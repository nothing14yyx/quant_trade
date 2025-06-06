# feature_selector.py
# 2025-06-03  完全替换旧版自动特征选择脚本
# 已修改要点：
#   1) 精简 orig_cols：不再把 interval/ignore/target_up 当作非特征列
#   2) 覆盖率过滤仅基于全表即可（保留原逻辑，但 orig_cols 中删掉无用列）
#   3) 交叉验证改为 TimeSeriesSplit，避免随机分层导致未来泄露
#   4) 在训练前剔除所有非数值列（尤其 datetime64），防止 LightGBM 报错

import yaml, json, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine

# ---------- 0. 读取配置 ----------
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

TARGET = cfg.get("feature_selector", {}).get("target", "target_down")
TOP_N  = cfg.get("feature_selector", {}).get("top_n", 12)
MIN_COVER = 0.10          # <10% 非空 → 丢弃
N_SPLIT   = 5

# ---------- 1. 取特征大表 ----------
df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time","close_time"])
# 原始列只留下真正不会出现在 all_features 的那些
orig_cols = {
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
    "symbol", TARGET  # 去掉 interval/ignore/target_up/target_down
}
all_features = [c for c in df.columns if c not in orig_cols]

# ---------- 2. 按周期归类 ----------
# 注意：此处只匹配 *_1h/*_4h/*_d1，不会误把 _isnan 或其他列包含进来
feature_pool = {
    "1h": [c for c in all_features if c.endswith("_1h") or c.endswith("_4h") or c.endswith("_d1")],
    "4h": [c for c in all_features if c.endswith("_4h") or c.endswith("_d1")],
    "1d": [c for c in all_features if c.endswith("_d1")],
}

yaml_out = {}

# ---------- 3. 周期循环 ----------
for period, cols in feature_pool.items():
    print(f"\n========== {period} 周期 ==========")
    use_cols = [c for c in cols if c in df.columns]

    # 3-1 去掉覆盖率过低列
    coverage = df[use_cols].notna().mean()
    keep_cols = coverage[coverage >= MIN_COVER].index.tolist()
    if not keep_cols:
        print("无有效特征列（全表覆盖率 < 10%），跳过该周期。")
        continue

    # ===== 新增：剔除所有非数值列（尤其 datetime64） =====
    numeric_cols = [
        c for c in keep_cols
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
    ]
    if not numeric_cols:
        print("剔除非数值列后没有剩余特征，跳过该周期。")
        continue
    keep_cols = numeric_cols

    # 3-2 留下目标非空行，做 TimeSeriesSplit 前要按时间排序
    subset = df[keep_cols + [TARGET, "open_time"]].dropna(subset=[TARGET])
    subset = subset.sort_values("open_time").reset_index(drop=True)

    if len(subset) < 800:
        print(f"样本太少（{len(subset)} < 800），跳过该周期。")
        continue

    X = subset[keep_cols]
    y = subset[TARGET].astype(int)
    times = subset["open_time"]

    # 3-3 TimeSeriesSplit：保证“训练集时间全在验证集时间之前”
    tscv = TimeSeriesSplit(n_splits=N_SPLIT)
    feat_imp = pd.Series(0.0, index=keep_cols)

    # 计算 pos_weight（在训练集里动态计算，避免用全量数据泄露）
    pos_weight_global = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight_global
    )

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X, y), 1):
        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

        # 重新计算每 fold 的 pos_weight，细粒度更好
        pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        lgb_params["scale_pos_weight"] = pw

        gbm = lgb.LGBMClassifier(**lgb_params)
        gbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        feat_imp += pd.Series(gbm.feature_importances_, index=keep_cols)

    feat_imp /= N_SPLIT
    feat_imp.sort_values(ascending=False, inplace=True)

    # 3-4 选 Top-N & 打印
    best_feats = feat_imp.head(TOP_N).index.tolist()
    print(f"Top-{TOP_N} 特征：")
    for f, sc in feat_imp.head(TOP_N).items():
        print(f"  {f:<40s}  {sc:.1f}")

    yaml_out[period] = best_feats

# ---------- 4. 保存 ----------
out_path = Path("utils/selected_features.yaml")
out_path.write_text(yaml.dump(yaml_out, allow_unicode=True))
print(f"\n✅ 已写入 {out_path.resolve()}")

# feature_selector.py
# 2025-06-03  完全替换旧版自动特征选择脚本
# ---------------------------------------------------------------

import yaml, json, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
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
MIN_COVER = 0.10          # <10 % 非空 → 丢弃
N_SPLIT   = 5

# ---------- 1. 取特征大表 ----------
df = pd.read_sql("SELECT * FROM features", engine)
orig_cols = {
    "open_time","open","high","low","close","volume","close_time",
    "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote",
    "symbol","interval","ignore",TARGET,"target_up","target_down"
}
all_features = [c for c in df.columns if c not in orig_cols]

# ---------- 2. 按周期归类 ----------
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
        print("无有效特征列，跳过。")
        continue

    subset = df[keep_cols + [TARGET]].dropna(subset=[TARGET])
    if len(subset) < 800:
        print(f"样本太少（{len(subset)}），跳过。")
        continue

    X = subset[keep_cols]
    y = subset[TARGET]

    # 3-2 LightGBM 五折交叉验证取平均重要度
    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=42)
    feat_imp = pd.Series(0.0, index=keep_cols)

    pos_weight = (y==0).sum() / max((y==1).sum(), 1)
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
        scale_pos_weight=pos_weight
    )

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        gbm = lgb.LGBMClassifier(**lgb_params)
        gbm.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[val], y.iloc[val])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        feat_imp += pd.Series(gbm.feature_importances_, index=keep_cols)

    feat_imp /= N_SPLIT
    feat_imp.sort_values(ascending=False, inplace=True)

    # 3-3 选 Top-N & 打印
    best_feats = feat_imp.head(TOP_N).index.tolist()
    print("Top-{} 特征:".format(TOP_N))
    for f, sc in feat_imp.head(TOP_N).items():
        print(f"  {f:<40s}  {sc:.1f}")

    yaml_out[period] = best_feats

# ---------- 4. 保存 ----------
out_path = Path("utils/selected_features.yaml")
out_path.write_text(yaml.dump(yaml_out, allow_unicode=True))
print(f"\n✅  已写入 {out_path.resolve()}")


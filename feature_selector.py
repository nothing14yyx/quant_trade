# feature_selector.py
# 2025-06-03  完全替换旧版自动特征选择脚本
# 已修改要点：
#   1) 精简 orig_cols：不再把 interval/ignore/target_up 当作非特征列
#   2) 覆盖率过滤仅基于全表即可（保留原逻辑，但 orig_cols 中删掉无用列）
#   3) 交叉验证改为 TimeSeriesSplit，避免随机分层导致未来泄露
#   4) 在训练前剔除所有非数值列（尤其 datetime64），防止 LightGBM 报错
#   5) 相关性阈值降至 0.85，并在此基础上计算 VIF，迭代剔除 VIF>10 的列
#      (VIF 计算时若数据行数太多，先抽样加速)

import os, yaml, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import shap
from sqlalchemy import create_engine
from statsmodels.stats.outliers_influence import variance_inflation_factor

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"

# ---------- 0. 读取配置 ----------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

TARGET = cfg.get("feature_selector", {}).get("target", "target_down")
TOP_N  = cfg.get("feature_selector", {}).get("top_n", 30)

# 与 FeatureEngineer 保持同步的未来字段列表，避免选入泄漏特征
FUTURE_COLS = [
    "future_volatility",
    "future_max_rise",
    "future_max_drawdown",
]

# 黑名单中默认加入 FUTURE_COLS，若 config 中已包含则去重
BLACKLIST = list({*cfg.get("feature_selector", {}).get("blacklist", []), *FUTURE_COLS})
MIN_COVER = 0.08          # <8% 非空 → 丢弃
ENABLE_PCA = False
PCA_COMPONENTS = 3
N_SPLIT   = 5
MAX_VIF_SAMPLE = 10000    # 计算 VIF 时最多抽样的行数

# ---------- 1. 取特征大表 ----------
df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time","close_time"])
# 原始列只留下真正不会出现在 all_features 的那些
orig_cols = {
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
    "symbol", TARGET, *FUTURE_COLS  # 去掉 interval/ignore/target_up/target_down
}
all_features = [
    c for c in df.columns
    if c not in orig_cols
    and c not in BLACKLIST
    and not c.endswith("_isnan")
]

# ---------- 2. 按周期归类 ----------
# 默认仅匹配 *_1h/*_4h/*_d1 结尾。为了让小时/周标记以及 _x/_y/_feat 列也能参与评估，
# 这里在后缀判断中额外加入這些情況。
base_suffixes = ("_1h", "_4h", "_d1")
time_cols = {"hour_of_day", "day_of_week"}
feature_pool = {
    "1h": [c for c in all_features if c.endswith(base_suffixes) or c in time_cols],
    "4h": [c for c in all_features if c.endswith(base_suffixes) or c in time_cols],
    "1d": [c for c in all_features if c.endswith(base_suffixes) or c in time_cols],
}

yaml_out = {}

# ---------- 3. 周期循环 ----------
for period, cols in feature_pool.items():
    print(f"\n========== {period} 周期 ==========")
    use_cols = [c for c in cols if c in df.columns]

    # ----- 根据周期过滤对应的时间行 -----
    if period == "4h":
        df_period = df[df["open_time"].dt.hour % 4 == 0]
    elif period == "1d":
        df_period = df[df["open_time"].dt.hour == 0]
    else:
        df_period = df

    # 3-1 去掉覆盖率过低列 (仅基于该周期数据)
    coverage = df_period[use_cols].notna().mean()
    keep_cols = coverage[coverage >= MIN_COVER].index.tolist()
    if not keep_cols:
        print("无有效特征列（覆盖率 < 5%），跳过该周期。")
        continue

    # ===== 新增：剔除所有非数值列（尤其 datetime64） =====
    numeric_cols = [
        c for c in keep_cols
        if pd.api.types.is_float_dtype(df_period[c]) or pd.api.types.is_integer_dtype(df_period[c])
    ]
    if not numeric_cols:
        print("剔除非数值列后没有剩余特征，跳过该周期。")
        continue
    keep_cols = numeric_cols

    # 3-2 留下目标非空行，做 TimeSeriesSplit 前要按时间排序
    subset = df_period[keep_cols + [TARGET, "open_time"]].dropna(subset=[TARGET])
    subset = subset.sort_values("open_time").reset_index(drop=True)

    if len(subset) < 800:
        print(f"样本太少（{len(subset)} < 800），跳过该周期。")
        continue

    if ENABLE_PCA and len(keep_cols) > 100:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset[keep_cols].fillna(0))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        comps = pca.fit_transform(X_scaled)
        for i in range(PCA_COMPONENTS):
            col = f"pca_{i+1}"
            subset[col] = comps[:, i]
            keep_cols.append(col)

    X = subset[keep_cols]
    y = subset[TARGET].astype(int)

    # 3-3 TimeSeriesSplit：保证“训练集时间全在验证集时间之前”
    tscv = TimeSeriesSplit(n_splits=N_SPLIT)
    shap_imp = pd.Series(0.0, index=keep_cols)

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
        # SHAP importance
        try:
            explainer = shap.TreeExplainer(gbm)
            sv = explainer.shap_values(X_val)
            if isinstance(sv, list):
                sv = sv[1]
            shap_imp += np.abs(sv).mean(0)
        except Exception as e:
            print("SHAP failed:", e)


    shap_imp /= N_SPLIT
    shap_rank = shap_imp.rank(pct=True)
    shap_rank.sort_values(ascending=False, inplace=True)

    cand_feats = shap_rank.head(TOP_N * 2).index.tolist()

    # 删除相关系数极高的特征，避免冗余
    corr = subset[cand_feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    vif_feats = [f for f in cand_feats if f not in to_drop]

    # 根据 VIF 进一步去除多重共线性
    while True:
        X_vif = subset[vif_feats].dropna()
        if len(X_vif) > MAX_VIF_SAMPLE:
            X_vif = X_vif.sample(MAX_VIF_SAMPLE, random_state=42)
        vifs = [variance_inflation_factor(X_vif[vif_feats].values, i)
                for i in range(len(vif_feats))]
        max_vif = max(vifs)
        if max_vif <= 10 or len(vif_feats) <= 1:
            break
        drop_idx = vifs.index(max_vif)
        dropped = vif_feats.pop(drop_idx)
        print(f"VIF {max_vif:.2f} -> 移除 {dropped}")

    final_feats = vif_feats[:TOP_N]

    print(f"Top-{TOP_N} 特征：")
    for f in final_feats:
        print(f"  {f:<40s}  {shap_rank[f]:.3f}")

    yaml_out[period] = final_feats

# ---------- 4. 保存 ----------
out_path = Path("utils/selected_features.yaml")
out_path.write_text(yaml.dump(yaml_out, allow_unicode=True))
print(f"\n✅ 已写入 {out_path.resolve()}")

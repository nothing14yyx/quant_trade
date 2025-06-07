# model_trainer.py  (2025-06-03, 已添加按 open_time 排序)
# ----------------------------------------------------------------
import yaml, joblib, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sqlalchemy import create_engine

# ---------- 1. 读取配置 ----------
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

# ---------- 2. 读取特征大表并按时间排序 ----------
df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time"])
# 确保整表按 open_time 升序排列，再 reset_index
df = df.sort_values("open_time").reset_index(drop=True)

# ---------- 3. 固定特征列 （来自 feature_selector 输出） ----------
feature_cols = {
    '1h': [
        'atr_pct_1h',              # 1h 波动率（ATR百分比）
        'rsi_slope_1h',            # 1h RSI斜率（动量变化）
        'kc_perc_1h',              # 1h Keltner通道分位（趋势/顺势）
        'vol_ma_ratio_1h',         # 1h 成交量/均线（量能）
        'boll_perc_1h',            # 1h 布林分位（价格偏离度）
        'fg_index',                # 日度情绪（恐惧贪婪）
        'funding_rate',            # 资金费率
        'cci_delta_1h',            # 1h CCI变化（顺势波动）
    ],
    '4h': [
        'atr_pct_4h',              # 4h 波动率
        'rsi_slope_4h',            # 4h RSI斜率
        'kc_perc_4h',              # 4h Keltner通道分位
        'vol_ma_ratio_4h',         # 4h 成交量/均线
        'boll_perc_4h',            # 4h 布林分位
        'fg_index_d1',             # 日度情绪（恐惧贪婪）
        'funding_rate_4h',         # 4h 资金费率
        'cci_delta_4h',            # 4h CCI变化
    ],
    'd1': [
        'atr_pct_d1',              # 日线波动率
        'rsi_slope_d1',            # 日线RSI斜率
        'kc_perc_d1',              # 日线Keltner通道分位
        'vol_ma_ratio_d1',         # 日线量能/均线
        'boll_perc_d1',            # 日线布林分位
        'fg_index_d1',             # 日线情绪
        'funding_rate_d1',         # 日线资金费率
        'cci_delta_d1',            # 日线CCI变化
    ],
}

# ---------- 4. 目标列 ----------
targets = {"up": "target_up", "down": "target_down"}

# ---------- 5. 训练函数 ----------
def train_one(df_all: pd.DataFrame,
              features: list[str],
              tgt: str,
              model_path: Path) -> None:

    # 5-1  缺列补 NaN
    for col in features:
        if col not in df_all.columns:
            df_all[col] = np.nan

    # 5-2  生成 NaN 标记列，并把标志加入特征列表
    feat_use = features.copy()
    for col in features:
        nan_flag = f"{col}_isnan"
        df_all[nan_flag] = df_all[col].isna().astype(int)
        feat_use.append(nan_flag)

    # 5-3  过滤掉标签为 NaN 的行后再取训练集
    data = df_all.dropna(subset=[tgt])
    # 此时 data 已经继承了原始 df_all 的顺序，且原始 df_all 事先已按 open_time 排序
    X = data[feat_use]
    y = data[tgt]

    # 5-4  计算类别不平衡补偿
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # 5-5  随机搜索 + 时间序列交叉验证（此处不再传 early-stopping callbacks）
    base = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=pos_weight,
        verbosity=-1,
    )

    param_grid = {
        "n_estimators": [300, 600, 900],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 20],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 0.1, 0.5],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_grid,
        n_iter=12,
        cv=tscv,
        scoring="roc_auc",
        random_state=42,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)

    best = search.best_estimator_

    # 5-6  用最后 20% 样本做验证，重新训练并 early-stop
    cut = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_val = y.iloc[:cut], y.iloc[cut:]

    best.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # 5-7  保存模型与特征列
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": feat_use},
                model_path, compress=3)
    print(f"✔ Saved: {model_path.name}  (CV-AUC {search.best_score_:.4f})")

    # 5-8  打印前 15 个重要特征
    imp = (pd.Series(best.feature_importances_, index=feat_use)
             .sort_values(ascending=False)
             .head(15))
    print(imp.to_string())

# ---------- 6. 周期 × 方向 训练循环 ----------
for period, cols in feature_cols.items():
    for tag, tgt_col in targets.items():
        print(f"\n🚀  Train {period}  {tag}")
        out_file = Path(f"models/model_{period}_{tag}.pkl")
        train_one(df, cols, tgt_col, out_file)

print("\n✅  All models finished.")

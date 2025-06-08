# model_trainer.py  (2025-06-03, 已添加按 open_time 排序)
# ----------------------------------------------------------------
import os, yaml, joblib, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from sqlalchemy import create_engine

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"

# ---------- 1. 读取配置 ----------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

# ---------- 1.1 训练选项 ----------
train_cfg = cfg.get("train_settings", {})
train_by_symbol = bool(train_cfg.get("by_symbol", False))
min_rows = int(train_cfg.get("min_rows", 500))
time_ranges = train_cfg.get("time_ranges", []) or [{"name": "all", "start": None, "end": None}]

# ---------- 2. 读取特征大表并按时间排序 ----------
df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time"])
# 确保整表按 open_time 升序排列，再 reset_index
df = df.sort_values("open_time").reset_index(drop=True)

# ---------- 3. 固定特征列 （来自 config.yaml 的 feature_cols） ----------
feature_cols = cfg.get("feature_cols", {})
if not feature_cols:
    raise RuntimeError("config.yaml 缺少 feature_cols 配置")

# ---------- 4. 目标列 ----------
targets = {"up": "target_up", "down": "target_down", "vol": "future_volatility"}

# ---------- 辅助：剔除极端异常样本 ----------
def drop_price_outliers(df: pd.DataFrame, pct: float = 0.995) -> pd.DataFrame:
    if not {"close", "open"}.issubset(df.columns):
        return df
    chg = (df["close"] / df["open"] - 1).abs()
    thresh = chg.quantile(pct)
    keep = chg <= thresh
    removed = len(df) - keep.sum()
    if removed:
        print(f"drop_price_outliers: removed {removed} rows")
    return df[keep]

# ---------- 5. 训练函数 ----------
def train_one(df_all: pd.DataFrame,
              features: list[str],
              tgt: str,
              model_path: Path,
              regression: bool = False) -> None:

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

    if regression:
        X = data[feat_use]
        y = data[tgt]
    else:
        pos_ratio = (data[tgt] == 1).mean()
        if pos_ratio < 0.4 or pos_ratio > 0.6:
            sampler = RandomOverSampler(random_state=42)
            res_X, res_y = sampler.fit_resample(data[feat_use + ["open_time"]], data[tgt])
            res = pd.DataFrame(res_X, columns=feat_use + ["open_time"])
            res[tgt] = res_y
            res = res.sort_values("open_time")
            X = res[feat_use]
            y = res[tgt]
        else:
            X = data[feat_use]
            y = data[tgt]

    # 5-4  计算类别不平衡补偿
    if not regression:
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # 5-5  随机搜索 + 时间序列交叉验证（此处不再传 early-stopping callbacks）
    if regression:
        base = lgb.LGBMRegressor(
            objective="regression",
            metric="l1",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
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
        scoring="neg_mean_absolute_error" if regression else "roc_auc",
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
        eval_metric="l1" if regression else "auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # 5-7  保存模型与特征列
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": feat_use},
                model_path, compress=3)
    score_label = "CV-MAE" if regression else "CV-AUC"
    print(f"✔ Saved: {model_path.name}  ({score_label} {search.best_score_:.4f})")

    # 5-8  打印前 15 个重要特征
    imp = (pd.Series(best.feature_importances_, index=feat_use)
             .sort_values(ascending=False)
             .head(15))
    print(imp.to_string())

# ---------- 6. 周期 × 方向 × 符号 训练循环 ----------
symbols = df["symbol"].unique() if train_by_symbol else [None]

for sym in symbols:
    df_sym = df[df["symbol"] == sym] if sym is not None else df
    if len(df_sym) < min_rows:
        continue
    for rng in time_ranges:
        df_rng = df_sym
        if rng.get("start"):
            df_rng = df_rng[df_rng["open_time"] >= pd.to_datetime(rng["start"])]
        if rng.get("end"):
            df_rng = df_rng[df_rng["open_time"] < pd.to_datetime(rng["end"])]
        df_rng = drop_price_outliers(df_rng)

        for period, cols in feature_cols.items():
            if period == "4h":
                subset = df_rng[df_rng["open_time"].dt.hour % 4 == 0]
            elif period in {"1d", "d1"}:
                subset = df_rng[df_rng["open_time"].dt.hour == 0]
            else:
                subset = df_rng

            for tag, tgt_col in targets.items():
                parts = [period]
                if sym is not None:
                    parts.append(sym)
                if rng.get("name") and rng["name"] != "all":
                    parts.append(rng["name"])
                parts.append(tag)
                file_name = "model_" + "_".join(parts) + ".pkl"
                print(f"\n🚀  Train {period} {sym or 'all'} {rng.get('name','all')} {tag}")
                out_file = Path("models") / file_name
                train_one(subset.copy(), cols, tgt_col, out_file, regression=(tag == "vol"))

print("\n✅  All models finished.")

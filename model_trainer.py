# model_trainer.py  (2025-06-03, 已添加按 open_time 排序)
# ----------------------------------------------------------------
import os
import re
import yaml
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


import optuna

logging.basicConfig(level=logging.INFO, format="%(message)s")
optuna.logging.set_verbosity(optuna.logging.ERROR)
from lightgbm.callback import CallbackEnv
from sqlalchemy import create_engine


def _sanitize_feature_names(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Sanitize feature names without mutating the original DataFrame."""

    mapping: dict[str, str] = {}
    sanitized: list[str] = []
    for col in features:
        clean = re.sub(r'[\[\]{}\\"\n\r\t]', "_", col)
        if clean != col:
            mapping[col] = clean
        sanitized.append(clean)
    df_out = df.rename(columns=mapping, inplace=False) if mapping else df
    return df_out, sanitized, mapping


def forward_chain_split(n_samples: int, n_splits: int = 5, gap: int = 0):
    """Yield train and validation indices in a forward chaining manner."""
    fold_size = n_samples // (n_splits + 1)
    indices = np.arange(n_samples)
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_start = train_end + gap
        val_end = val_start + fold_size
        if val_end > n_samples:
            val_end = n_samples
        yield indices[:train_end], indices[val_start:val_end]


# ---------- 自定义：只在过去样本内合成的 SMOTE ----------
class TimeSeriesAwareSMOTE:
    """在时间序列数据上执行 SMOTE，支持按时间分组，确保仅使用过去的少数类样本"""

    def __init__(
        self,
        k_neighbors: int = 2,
        random_state: int | None = 42,
        group_freq: str | None = None,
    ):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.group_freq = group_freq
        self.sample_indices_: list[int] | None = None

    def _resample_one(self, X, y, open_time):
        rng = np.random.RandomState(self.random_state)
        minority = y.value_counts().idxmin()
        maj_count = (y != minority).sum()
        min_idx = np.where(y == minority)[0]
        deficit = maj_count - len(min_idx)
        if deficit <= 0:
            return X, y, np.arange(len(X))

        new_rows = []
        new_times = []
        sample_idx = list(range(len(X)))
        for idx in min_idx:
            prev_idx = min_idx[min_idx < idx]
            if len(prev_idx) == 0:
                continue
            neighbor = rng.choice(prev_idx)
            alpha = rng.rand()
            row = X.iloc[idx] + alpha * (X.iloc[neighbor] - X.iloc[idx])
            new_rows.append(row)
            new_times.append(open_time.iloc[idx])
            sample_idx.append(idx)
            if len(new_rows) >= deficit:
                break

        if not new_rows:
            return X, y, np.arange(len(X))

        X_aug = pd.concat([X, pd.DataFrame(new_rows)], ignore_index=True)
        y_aug = pd.concat([y, pd.Series([minority] * len(new_rows))], ignore_index=True)
        time_aug = pd.concat([open_time, pd.Series(new_times)], ignore_index=True)

        order = np.argsort(time_aug)
        return (
            X_aug.iloc[order].reset_index(drop=True),
            y_aug.iloc[order].reset_index(drop=True),
            np.array(sample_idx)[order],
        )

    def fit_resample(self, X: pd.DataFrame, y: pd.Series, open_time: pd.Series):
        if self.group_freq:
            groups = open_time.dt.to_period(self.group_freq)
            res_X_all, res_y_all, idx_all = [], [], []
            for g in groups.unique():
                mask = groups == g
                X_res, y_res, idx_res = self._resample_one(
                    X[mask].reset_index(drop=True),
                    y[mask].reset_index(drop=True),
                    open_time[mask].reset_index(drop=True),
                )
                offset = len(pd.concat(res_X_all)) if res_X_all else 0
                idx_all.extend((np.array(idx_res) + offset).tolist())
                res_X_all.append(X_res)
                res_y_all.append(y_res)
            X_out = pd.concat(res_X_all, ignore_index=True)
            y_out = pd.concat(res_y_all, ignore_index=True)
            self.sample_indices_ = np.array(idx_all)
            return X_out, y_out
        else:
            X_res, y_res, idx_res = self._resample_one(X, y, open_time)
            self.sample_indices_ = idx_res
            return X_res, y_res


class OffsetLightGBMPruningCallback:
    """在交叉验证中为 LightGBM 的每折评估添加 step 偏移，避免 Optuna 重复 step 警告"""

    def __init__(
        self,
        trial: optuna.Trial,
        metric: str,
        valid_name: str = "valid_0",
        report_interval: int = 1,
        step_offset: int = 0,
    ) -> None:
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._report_interval = report_interval
        self._step_offset = step_offset

    def __call__(self, env: CallbackEnv) -> None:
        if (env.iteration + 1) % self._report_interval != 0:
            return

        evals = env.evaluation_result_list
        if not evals:
            return

        is_cv = len(evals[0]) == 5
        target_valid_name = "valid" if is_cv else self._valid_name

        for valid_name, metric, value in [(e[0], e[1], e[2]) for e in evals]:
            if valid_name == target_valid_name and (
                metric == self._metric or metric == "valid " + self._metric
            ):
                step = env.iteration + self._step_offset
                self._trial.report(value, step=step)
                if self._trial.should_prune():
                    raise optuna.TrialPruned(f"Trial was pruned at iteration {step}.")
                break
        else:
            raise ValueError(
                f'The entry associated with the validation name "{target_valid_name}" '
                f'and the metric name "{self._metric}" is not found in the evaluation result list '
                f"{str(evals)}."
            )


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
time_ranges = train_cfg.get("time_ranges", []) or [
    {"name": "all", "start": None, "end": None}
]
use_ts_smote = bool(train_cfg.get("ts_smote", False))
ts_smote_group_freq = train_cfg.get("ts_smote_group_freq")
hold_days = int(train_cfg.get("hold_days", 0))
n_trials = int(train_cfg.get("n_trials", 40))

# ---------- 2. 读取特征大表并按时间排序 ----------
df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time"])
# 确保整表按 open_time 升序排列，再 reset_index
df = df.sort_values("open_time").reset_index(drop=True)

# ---------- 3. 固定特征列 （来自 config.yaml 的 feature_cols） ----------
feature_cols = cfg.get("feature_cols", {})
if not feature_cols:
    raise RuntimeError("config.yaml 缺少 feature_cols 配置")
# 将 1d 统一映射为 d1，避免重复训练
if "1d" in feature_cols:
    feature_cols["d1"] = feature_cols.pop("1d")

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
        logging.info(f"drop_price_outliers: removed {removed} rows")
    return df[keep]


# ---------- 5. 训练函数 ----------
def train_one(
    df_all: pd.DataFrame,
    features: list[str],
    tgt: str,
    model_path: Path,
    period: str,
    tag: str,
    regression: bool = False,
) -> None:

    # 5-1  缺列补 NaN，并记录重命名映射
    df_all, feat_use, rename_map = _sanitize_feature_names(df_all, features.copy())

    for col in feat_use:
        if col not in df_all.columns:
            df_all[col] = np.nan

    # 5-2  不再额外拼接 _isnan 标记列，避免重复

    # 5-3  过滤掉标签为 NaN 的行后再取训练集
    data = df_all.dropna(subset=[tgt])
    # 此时 data 已经继承了原始 df_all 的顺序，且原始 df_all 事先已按 open_time 排序

    X = data[feat_use]
    y = data[tgt]

    # ---- 留出集
    if hold_days > 0:
        end_time = data["open_time"].max()
        hold_mask = data["open_time"] >= end_time - pd.Timedelta(days=hold_days)
        hold_df = data[hold_mask]
        data = data[~hold_mask]
        X_hold = hold_df[feat_use]
        y_hold = hold_df[tgt]
    else:
        X_hold = pd.DataFrame()
        y_hold = pd.Series(dtype=y.dtype)
    X = data[feat_use]
    y = data[tgt]

    # 时序切分
    splits = list(forward_chain_split(len(X), n_splits=5))

    imputer = SimpleImputer(strategy="median")

    def objective(trial: optuna.Trial):
        if period == "d1":
            params = {
                "learning_rate": trial.suggest_float("lr_d1", 0.01, 0.03, log=True),
                "num_leaves": trial.suggest_int("nl_d1", 64, 160, step=8),
                "n_estimators": trial.suggest_int("ne_d1", 600, 1200, step=100),
                "max_depth": trial.suggest_int("md_d1", 4, 12),
                "min_child_samples": trial.suggest_int("mcs_d1", 20, 100),
                "subsample": trial.suggest_float("subsample_d1", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("cbt_d1", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda_d1", 0.0, 0.5),
            }
        else:
            params = {
                "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("nl", 31, 127),
                "n_estimators": trial.suggest_int("ne", 300, 900),
                "max_depth": trial.suggest_int("md", -1, 20),
                "min_child_samples": trial.suggest_int("mcs", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("cbt", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            }

        n_boost_rounds = params.get("n_estimators", params.get("ne", params.get("ne_d1")))
        fold_scores: list[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)

            extra_params: dict[str, float] = {}
            if regression:
                X_res_imp, y_res = X_tr_imp, y_tr
            elif use_ts_smote:
                smote = TimeSeriesAwareSMOTE(group_freq=ts_smote_group_freq)
                X_res_imp, y_res = smote.fit_resample(
                    X_tr_imp,
                    y_tr,
                    data.loc[X_tr.index, "open_time"],
                )
            else:
                X_res_imp, y_res = X_tr_imp, y_tr
                pos_weight = min((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 50)
                extra_params["scale_pos_weight"] = pos_weight

            if regression:
                model = lgb.LGBMRegressor(
                    objective="regression",
                    metric="l1",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                    **params,
                )
            else:
                model = lgb.LGBMClassifier(
                    objective="binary",
                    metric="auc",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                    **extra_params,
                    **params,
                )

            model.fit(
                X_res_imp,
                y_res,
                eval_set=[(X_va_imp, y_va)],
                eval_metric="l1" if regression else "auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    OffsetLightGBMPruningCallback(
                        trial,
                        "l1" if regression else "auc",
                        report_interval=10,
                        step_offset=fold_i * n_boost_rounds,
                    ),
                ],
            )

            if regression:
                preds = model.predict(X_va_imp, num_iteration=model.best_iteration_)
                fold_scores.append(-mean_absolute_error(y_va, preds))
            else:
                preds = model.predict_proba(X_va_imp, num_iteration=model.best_iteration_)[:, 1]
                fold_scores.append(roc_auc_score(y_va, preds))

        return float(np.mean(fold_scores))

    # 5-5  使用 Optuna + pruner 进行超参搜索

    if period == "d1":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5000, reduction_factor=4)
    elif period == "4h":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1500)
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2000)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_params = study.best_params
    if period == "d1":
        best_params = {
            "learning_rate": raw_params["lr_d1"],
            "num_leaves": raw_params["nl_d1"],
            "n_estimators": raw_params["ne_d1"],
            "max_depth": raw_params["md_d1"],
            "min_child_samples": raw_params["mcs_d1"],
            "subsample": raw_params["subsample_d1"],
            "colsample_bytree": raw_params["cbt_d1"],
            "reg_lambda": raw_params["reg_lambda_d1"],
        }
    else:
        best_params = {
            "learning_rate": raw_params["lr"],
            "num_leaves": raw_params["nl"],
            "n_estimators": raw_params["ne"],
            "max_depth": raw_params["md"],
            "min_child_samples": raw_params["mcs"],
            "subsample": raw_params["subsample"],
            "colsample_bytree": raw_params["cbt"],
            "reg_lambda": raw_params["reg_lambda"],
        }

    # ----- 依据最后一折重新训练最佳模型 -----
    _, va_idx = splits[-1]
    X_all_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_va_imp = X_all_imp.iloc[va_idx]
    y_va = y.iloc[va_idx]

    extra_params: dict[str, float] = {}
    X_res_imp, y_res = X_all_imp, y
    if not regression:
        pos_weight = min((y == 0).sum() / max((y == 1).sum(), 1), 50)
        extra_params["scale_pos_weight"] = pos_weight

    if regression:
        best = lgb.LGBMRegressor(
            objective="regression",
            metric="l1",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **best_params,
        )
    else:
        best = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **extra_params,
            **best_params,
        )

    best.fit(
        X_res_imp,
        y_res,
        eval_set=[(X_va_imp, y_va)],
        eval_metric="l1" if regression else "auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # ----- 留出集评估 -----
    if len(X_hold):
        X_hold_imp = pd.DataFrame(imputer.transform(X_hold), columns=X_hold.columns, index=X_hold.index)
        if regression:
            hold_pred = best.predict(X_hold_imp, num_iteration=best.best_iteration_)
            hold_score = mean_absolute_error(y_hold, hold_pred)
        else:
            hold_pred = best.predict_proba(X_hold_imp, num_iteration=best.best_iteration_)[
                :, 1
            ]
            hold_score = roc_auc_score(y_hold, hold_pred)
        label = "Holdout-MAE" if regression else "Holdout-AUC"
        logging.info(f"{label}: {hold_score:.4f}")

    feat_imp = getattr(best, "feature_importances_", None)

    # ----- 根据验证集寻找最佳阈值 -----
    best_th = 0.5
    if not regression:
        val_proba = best.predict_proba(X_va_imp, num_iteration=best.best_iteration_)[:, 1]
        prec, rec, thr = precision_recall_curve(y_va, val_proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if len(thr):
            best_th = float(thr[np.nanargmax(f1)])

    # 5-7  保存模型与特征列及阈值
    pipe = Pipeline([
        ("imputer", imputer),
        ("model", best),
    ])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features": feat_use,
            "threshold": best_th,
            "rename_map": rename_map,
            "sampled": bool(use_ts_smote),
            "version": datetime.datetime.utcnow().strftime("%Y%m%d"),
        },
        model_path,
        compress=3,
    )
    study.trials_dataframe().to_csv(model_path.parent / "optuna_trials.csv", index=False)
    score_label = "CV-MAE" if regression else "CV-AUC"
    best_val = abs(study.best_value) if regression else study.best_value
    logging.info(
        f"✔ Saved: {model_path.name}  ({score_label} {best_val:.4f}, th{best_th:.3f})"
    )

    # 5-8  打印前 15 个重要特征
    if feat_imp is not None:
        imp = pd.Series(feat_imp, index=feat_use).sort_values(ascending=False)
        imp.to_csv(model_path.with_suffix(".feat_imp.csv"))
        logging.info(imp.head(15).to_string())


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
            elif period == "d1":
                subset = df_rng[df_rng["open_time"].dt.hour == 0]
            else:
                subset = df_rng

            for tag, tgt_col in targets.items():
                file_name = f"model_{period}_{tag}.pkl"
                logging.info(
                    f"\n🚀  Train {period} {sym or 'all'} {rng.get('name','all')} {tag}"
                )
                out_dir = Path("models") / sym if sym is not None else Path("models")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / file_name
                train_one(
                    subset.copy(),
                    cols,
                    tgt_col,
                    out_file,
                    period,
                    tag,
                    regression=(tag == "vol"),
                )

logging.info("\n✅  All models finished.")

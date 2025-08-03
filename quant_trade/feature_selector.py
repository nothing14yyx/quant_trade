# feature_selector.py
# 2025-06-03  完全替换旧版自动特征选择脚本
# 已修改要点：
#   1) 精简 orig_cols：不再把 interval/ignore/target 当作非特征列
#   2) 覆盖率过滤仅基于全表即可（保留原逻辑，但 orig_cols 中删掉无用列）
#   3) 交叉验证改为 TimeSeriesSplit，避免随机分层导致未来泄露
#   4) 在训练前剔除所有非数值列（尤其 datetime64），防止 LightGBM 报错
#   5) 相关性阈值降至 0.85，并在此基础上计算 VIF，迭代剔除 VIF>10 的列
#      (VIF 计算时若数据行数太多，先抽样加速)

import os
import yaml
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import shap
import logging
from sqlalchemy import create_engine
from statsmodels.stats.outliers_influence import variance_inflation_factor

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"
logger = logging.getLogger(__name__)

# ---------- 0. 读取配置 ----------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

fs_cfg = cfg.get("feature_selector", {})
targets = fs_cfg.get("targets") or fs_cfg.get("target", "target")
if not isinstance(targets, list):
    targets = [targets]
ROWS = fs_cfg.get("rows")
START_TIME = fs_cfg.get("start_time")
TOP_N = fs_cfg.get("top_n", 30)
CORR_THRESH = fs_cfg.get("corr_thresh", 0.85)
MAX_VIF = fs_cfg.get("max_vif", 10)
MIN_COVER_MAP = fs_cfg.get("min_cover_map", {"1h": 0.8, "4h": 0.7, "d1": 0.6})
EARLY_STOP = fs_cfg.get("early_stopping_rounds", 30)
USE_PERM = fs_cfg.get("use_permutation", False)
VAR_THRESH = fs_cfg.get("var_thresh", 1e-5)
ROWS_LIMIT = fs_cfg.get("rows")

logger.info(
    "Config loaded: corr_thresh=%s, max_vif=%s, min_cover_map=%s, early_stop=%s, use_permutation=%s, var_thresh=%s, rows=%s, start_time=%s",
    CORR_THRESH,
    MAX_VIF,
    MIN_COVER_MAP,
    EARLY_STOP,
    USE_PERM,
    VAR_THRESH,
    ROWS,
    START_TIME,
)

# 与 FeatureEngineer 保持同步的未来字段列表，避免选入泄漏特征
FUTURE_COLS = [
    "future_volatility",
    "future_volatility_1h",
    "future_volatility_4h",
    "future_volatility_d1",
    "future_max_rise",
    "future_max_rise_1h",
    "future_max_rise_4h",
    "future_max_rise_d1",
    "future_max_drawdown",
    "future_max_drawdown_1h",
    "future_max_drawdown_4h",
    "future_max_drawdown_d1",
    "target",
    "target_1h",
    "target_4h",
    "target_d1",
]

# 黑名单中默认加入 FUTURE_COLS，若 config 中已包含则去重
BLACKLIST = list({*cfg.get("feature_selector", {}).get("blacklist", []), *FUTURE_COLS})
ENABLE_PCA = False
PCA_COMPONENTS = 3
N_SPLIT   = 5
MAX_VIF_SAMPLE = 10000    # 计算 VIF 时最多抽样的行数

# ---------- 1. 数据加载辅助函数 ----------

def load_feature_data(
    *, start_time: pd.Timestamp | str | None = None, rows: int | None = None
) -> pd.DataFrame:
    """从数据库读取 ``features`` 表数据，可选按时间或行数过滤"""
    query = "SELECT * FROM features"
    params: dict[str, object] | None = None
    if start_time is not None:
        query += " WHERE open_time >= %(start)s"
        params = {"start": pd.to_datetime(start_time)}
    query += " ORDER BY open_time"
    if rows:
        query += f" LIMIT {rows}"
    return pd.read_sql(
        query,
        engine,
        params=params,
        parse_dates=["open_time", "close_time"],
    )


def select_features(
    target: str,
    df: pd.DataFrame | None = None,
    corr_thresh: float = CORR_THRESH,
    max_vif: float = MAX_VIF,
    min_cover_map: dict[str, float] = MIN_COVER_MAP,
    early_stopping_rounds: int = EARLY_STOP,
    var_thresh: float = VAR_THRESH,
    use_permutation: bool = USE_PERM,
    rows: int | None = ROWS_LIMIT,
) -> None:
    if df is None:
        df = load_feature_data(start_time=START_TIME, rows=ROWS)
    orig_cols = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "symbol",
        target,
        *FUTURE_COLS,
    }

    all_features = [
        c
        for c in df.columns
        if c not in orig_cols and c not in BLACKLIST
    ]

    time_cols = {"hour_of_day", "day_of_week"}
    feature_pool = {
        "1h": [c for c in all_features if "_1h" in c or c in time_cols],
        "4h": [c for c in all_features if "_4h" in c or c in time_cols],
        "d1": [c for c in all_features if "_d1" in c or c in time_cols],
    }

    yaml_out = {}

    for period, cols in feature_pool.items():
        logger.info("========== %s 周期 ==========", period)
        use_cols = [c for c in cols if c in df.columns]

        if period == "4h":
            df_period = df[df["open_time"].dt.hour % 4 == 0]
        elif period == "d1":
            df_period = df[df["open_time"].dt.hour == 0]
        else:
            df_period = df

        coverage = df_period[use_cols].notna().mean()
        min_cover = float(min_cover_map.get(period, 0))
        keep_cols = coverage[coverage >= min_cover].index.tolist()
        if not keep_cols:
            logger.info("无有效特征列（覆盖率 < %.0f%%），跳过该周期。", min_cover * 100)
            continue

        numeric_cols = [
            c
            for c in keep_cols
            if pd.api.types.is_float_dtype(df_period[c])
            or pd.api.types.is_integer_dtype(df_period[c])
        ]
        if not numeric_cols:
            logger.info("剔除非数值列后没有剩余特征，跳过该周期。")
            continue
        keep_cols = numeric_cols

        subset = df_period[keep_cols + [target, "open_time"]].dropna(subset=[target])
        subset = subset.sort_values("open_time").reset_index(drop=True)

        if len(subset) < 800:
            logger.info("样本太少（%s < 800），跳过该周期。", len(subset))
            continue

        variances = subset[keep_cols].var()
        keep_cols = variances[variances >= var_thresh].index.tolist()
        if not keep_cols:
            logger.info("低方差过滤后没有剩余特征，跳过该周期。")
            continue

        if ENABLE_PCA and len(keep_cols) > 100:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(subset[keep_cols])
            from sklearn.decomposition import PCA

            pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
            comps = pca.fit_transform(X_scaled)
            for i in range(PCA_COMPONENTS):
                col = f"pca_{i+1}"
                subset[col] = comps[:, i]
                keep_cols.append(col)

        X = subset[keep_cols]
        y = subset[target]

        is_cls = target.startswith("target")
        if is_cls and y.nunique() < 2:
            logger.info("标签只有一个类别，跳过该周期。")
            continue

        tscv = TimeSeriesSplit(n_splits=N_SPLIT)
        shap_imp = pd.Series(0.0, index=keep_cols, dtype=float)
        perm_imp = pd.Series(0.0, index=keep_cols, dtype=float) if use_permutation else None
        fold_cnt = 0

        if is_cls:
            lgb_params = dict(
                objective="multiclass",
                num_class=3,
                metric="multi_logloss",
                learning_rate=0.05,
                num_leaves=64,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=1000,
                random_state=42,
                n_jobs=-1,
            )
        else:
            lgb_params = dict(
                objective="regression",
                metric="l1",
                learning_rate=0.05,
                num_leaves=64,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=1000,
                random_state=42,
                n_jobs=-1,
            )

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X, y), 1):
            X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            if is_cls and y_train.nunique() < 2:
                logger.info("Fold %s 标签只有一个类别，跳过当前折。", fold)
                continue

            if is_cls:
                gbm = lgb.LGBMClassifier(**lgb_params)
                eval_metric = ["multi_logloss", "multi_error"]
                metric_key = "multi_logloss"
            else:
                gbm = lgb.LGBMRegressor(**lgb_params)
                eval_metric = "l1"
                metric_key = "l1"

            gbm.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=eval_metric,
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            )
            score = gbm.best_score_["valid_0"][metric_key]
            logger.info("Fold %s %s: %.4f", fold, metric_key, score)
            fold_cnt += 1
            try:
                explainer = shap.TreeExplainer(gbm)
                sv = explainer.shap_values(X_val)
                if isinstance(sv, list):
                    sv = np.mean([np.abs(s) for s in sv], axis=0)
                shap_imp += np.abs(sv).mean(0)
            except (ValueError, RuntimeError) as exc:
                logger.exception("SHAP failed: %s", exc)
                shap_imp += pd.Series(gbm.feature_importances_, index=keep_cols)

            if use_permutation:
                from sklearn.inspection import permutation_importance
                scoring = "neg_log_loss" if is_cls else "neg_mean_absolute_error"
                perm_res = permutation_importance(
                    gbm,
                    X_val,
                    y_val,
                    n_repeats=5,
                    random_state=42,
                    scoring=scoring,
                )
                perm_imp += perm_res.importances_mean

        if fold_cnt == 0:
            logger.info("无有效折，跳过该周期。")
            continue

        shap_imp /= fold_cnt
        if use_permutation:
            perm_imp /= fold_cnt
        shap_rank = shap_imp.rank(pct=True)
        shap_rank.sort_values(ascending=False, inplace=True)
        if use_permutation:
            perm_series = pd.Series(perm_imp, index=keep_cols)
            perm_rank = perm_series.rank(pct=True)
            perm_rank.sort_values(ascending=False, inplace=True)

        cand_feats = shap_rank.head(TOP_N * 2).index.tolist()

        corr = subset[cand_feats].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
        vif_feats = [f for f in cand_feats if f not in to_drop]

        while True:
            X_vif = subset[vif_feats].dropna()
            if len(X_vif) > MAX_VIF_SAMPLE:
                X_vif = X_vif.sample(MAX_VIF_SAMPLE, random_state=42)
            vifs = [variance_inflation_factor(X_vif[vif_feats].values, i) for i in range(len(vif_feats))]
            max_vif_val = max(vifs)
            if max_vif_val <= max_vif or len(vif_feats) <= 1:
                break
            drop_idx = vifs.index(max_vif_val)
            dropped = vif_feats.pop(drop_idx)
            logger.info("VIF %.2f -> 移除 %s", max_vif_val, dropped)

        final_feats = vif_feats[:TOP_N]

        logger.info("Top-%s 特征：", TOP_N)
        for f in final_feats:
            logger.info("  %-40s  %.3f", f, shap_rank[f])
        if use_permutation:
            logger.info("Permutation Top-%s：", TOP_N)
            for f in perm_rank.head(TOP_N).index:
                logger.info("  %-40s  %.3f", f, perm_rank[f])

        yaml_out[period] = final_feats

    out_path = Path("selected_features") / f"selected_features_{target.replace('target_', '')}.yaml"
    out_path.write_text(yaml.dump(yaml_out, allow_unicode=True))
    logger.info("\n✅ 已写入 %s", out_path.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_feature_data(start_time=START_TIME, rows=ROWS)
    for t in targets:
        logger.info("\n====== 处理 %s ======", t)
        select_features(t, df)


def update_selected_features(
    df: pd.DataFrame | None,
    period: str,
    target: str,
    yaml_file: Path | str = Path("selected_features/selected_features.yaml"),
    shap_thresh: float = 0.01,
    ic_thresh: float = 0.01,
    rows: int | None = ROWS_LIMIT,
) -> list[str]:
    """根据最近样本的 SHAP 值和因子 IC 更新特征列表

    Parameters
    ----------
    df : pandas.DataFrame or None
        含最新特征数据与目标列的表格，若为 ``None`` 则自动从数据库读取。
    period : str
        周期名称，如 ``"1h"``、``"4h"``。
    target : str
        目标列名。
    yaml_file : Path or str, default ``"selected_features/selected_features.yaml"``
        保存特征列表的 YAML 文件路径。
    shap_thresh : float, default 0.01
        相对平均 SHAP 值低于该阈值的特征将被移除。
    ic_thresh : float, default 0.01
        绝对因子 IC 低于该阈值的特征将被移除。

    Returns
    -------
    list[str]
        更新后的特征列表。
    """
    if df is None:
        df = load_feature_data(start_time=START_TIME, rows=ROWS)

    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        logger.warning("%s 不存在", yaml_path)
        return []

    config = yaml.safe_load(yaml_path.read_text()) or {}
    feats = config.get(period, [])
    if not feats:
        logger.warning("%s 周期无已选特征", period)
        return []

    subset = df.dropna(subset=[target])
    subset = subset.sort_values("open_time").reset_index(drop=True)
    subset = subset[feats + [target]].dropna()
    if len(subset) < 10:
        logger.warning("样本不足，保留原列表")
        return feats

    X = subset[feats]
    y = subset[target]

    mdl = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    )
    mdl.fit(X, y)

    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = np.mean([np.abs(s) for s in sv], axis=0)
    shap_imp = np.abs(sv).mean(0)
    shap_imp /= shap_imp.sum() or 1.0

    ic_vals = {c: float(X[c].corr(y)) for c in feats}
    keep = [
        c
        for c, imp in zip(feats, shap_imp)
        if imp >= shap_thresh and abs(ic_vals.get(c, 0.0)) >= ic_thresh
    ]

    config[period] = keep
    yaml_path.write_text(yaml.dump(config, allow_unicode=True))
    logger.info("已更新 %s 的特征列表，共 %s 项", period, len(keep))
    return keep



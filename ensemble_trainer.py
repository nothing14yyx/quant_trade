import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sqlalchemy import create_engine

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None  # optional dependency

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}?charset=utf8mb4"
)

feature_cols = cfg.get("feature_cols", {})
if not feature_cols:
    raise RuntimeError("config.yaml 缺少 feature_cols 配置")

targets = {"up": "target_up", "down": "target_down"}


def train_ensemble(df_all: pd.DataFrame, features: list[str], tgt: str, model_path: Path) -> None:
    for col in features:
        if col not in df_all.columns:
            df_all[col] = np.nan

    feat_use = features.copy()
    for col in features:
        nan_flag = f"{col}_isnan"
        df_all[nan_flag] = df_all[col].isna().astype(int)
        feat_use.append(nan_flag)

    data = df_all.dropna(subset=[tgt])
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

    base_models = []
    base_models.append(("lgb", lgb.LGBMClassifier(objective="binary", random_state=42, n_jobs=1)))
    base_models.append(("xgb", xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=200, eval_metric="auc")))
    if CatBoostClassifier is not None:
        base_models.append(("cat", CatBoostClassifier(depth=6, learning_rate=0.1, verbose=False)))
    base_models.append(("mlp", MLPClassifier(max_iter=300, random_state=42)))
    if TabNetClassifier is not None:
        base_models.append(("tabnet", TabNetClassifier(verbose=0)))

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=200),
        passthrough=True,
        n_jobs=-1,
    )

    param_grid = {
        "lgb__n_estimators": [300, 600, 900, 1200],
        "lgb__num_leaves": [31, 63, 127],
        "xgb__max_depth": [3, 6, 9],
        "xgb__learning_rate": [0.03, 0.1, 0.2],
        "mlp__hidden_layer_sizes": [(64,), (128,), (256,)],
    }
    if CatBoostClassifier is not None:
        param_grid["cat__depth"] = [6, 8, 10]
        param_grid["cat__learning_rate"] = [0.05, 0.1, 0.2]
    if TabNetClassifier is not None:
        param_grid["tabnet__n_d"] = [8, 16, 32]
        param_grid["tabnet__n_steps"] = [3, 5, 7]

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=stack,
        param_distributions=param_grid,
        n_iter=24,
        cv=tscv,
        scoring="roc_auc",
        random_state=42,
        verbose=1,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X, y)
    best = search.best_estimator_

    # finer search around best
    fine_grid = {
        "lgb__n_estimators": [best.named_estimators_["lgb"].n_estimators + i for i in (-100, 0, 100)],
        "xgb__max_depth": [best.named_estimators_["xgb"].max_depth + i for i in (-1, 0, 1)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
    }
    search2 = RandomizedSearchCV(
        estimator=best,
        param_distributions=fine_grid,
        n_iter=12,
        cv=tscv,
        scoring="roc_auc",
        random_state=7,
        verbose=1,
        refit=True,
        n_jobs=-1,
    )
    search2.fit(X, y)
    final_model = search2.best_estimator_

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": feat_use}, model_path, compress=3)
    print(f"✔ Saved ensemble model: {model_path.name}")


def main() -> None:
    df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time"])  # noqa
    df = df.sort_values("open_time").reset_index(drop=True)

    for period, cols in feature_cols.items():
        for tag, tgt_col in targets.items():
            file_name = f"ensemble_{period}_{tag}.pkl"
            print(f"\n🚀 Train ensemble {period} {tag}")
            out_file = Path("models") / file_name
            train_ensemble(df.copy(), cols, tgt_col, out_file)

    print("\n✅ All ensemble models finished.")


if __name__ == "__main__":
    main()

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class AIModelPredictor:
    """封装AI模型的加载与推理"""

    def __init__(self, model_paths: dict[str, dict[str, str]]):
        self.models: dict[str, dict[str, dict[str, Any]]] = {}
        self.calibrators: dict[str, dict[str, Any]] = {}
        base_dir = Path(__file__).resolve().parent

        allowed = {"1h", "4h"}
        for period, path_dict in model_paths.items():
            if period not in allowed:
                continue

            self.models[period] = {}
            self.calibrators[period] = {}
            for direction, path in path_dict.items():
                p = Path(path)
                if not p.is_absolute():
                    p = base_dir / p
                loaded = joblib.load(p)
                pipe = loaded["pipeline"]
                if hasattr(pipe, "set_output"):
                    pipe.set_output(transform="pandas")
                self.models[period][direction] = {
                    "pipeline": loaded["pipeline"],
                    "features": loaded["features"],
                }
                self.calibrators[period][direction] = loaded.get("calibrator")

    def _build_df(self, features, train_cols: list[str]) -> pd.DataFrame:
        """构建 pandas ``DataFrame`` 以便送入模型.

        ``features`` 可以是单条特征 ``Mapping``、``list``[``Mapping``]
        或者二维 ``ndarray``/``list``，以适配批量推理需求。
        """

        if isinstance(features, Mapping):
            row = {c: [features.get(c, np.nan)] for c in train_cols}
            df = pd.DataFrame(row)
        elif isinstance(features, (list, tuple)) and features and isinstance(features[0], Mapping):
            data = [{c: f.get(c, np.nan) for c in train_cols} for f in features]
            df = pd.DataFrame(data, columns=train_cols)
        else:
            arr = np.asarray(features)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            df = pd.DataFrame(arr, columns=train_cols)

        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return df

    def predict_proba(
        self, period: str, direction: str, features: Mapping[str, float | int | None]
    ) -> float:
        model = self.models[period][direction]
        feature_cols = model["features"]
        pipeline = model["pipeline"]
        df = self._build_df(features, feature_cols)
        proba = pipeline.predict_proba(df)[:, 1]
        calibrator = self.calibrators[period].get(direction)
        if calibrator is not None:
            proba = calibrator.transform(proba.reshape(-1, 1)).ravel()
        return float(proba[0])

    def get_ai_score(
        self,
        features,
        model_up: Mapping[str, Any],
        model_down: Mapping[str, Any],
        calibrator_up: Any | None = None,
        calibrator_down: Any | None = None,
    ) -> float | np.ndarray:
        """根据上涨/下跌模型概率差值计算AI得分"""
        X_up = self._build_df(features, model_up["features"])
        X_down = self._build_df(features, model_down["features"])
        prob_up = model_up["pipeline"].predict_proba(X_up)[:, 1]
        prob_down = model_down["pipeline"].predict_proba(X_down)[:, 1]
        if calibrator_up is not None:
            prob_up = calibrator_up.transform(prob_up.reshape(-1, 1)).ravel()
        if calibrator_down is not None:
            prob_down = calibrator_down.transform(prob_down.reshape(-1, 1)).ravel()
        denom = prob_up + prob_down
        ai_score = np.where(denom == 0, 0.0, (prob_up - prob_down) / denom)
        if ai_score.size == 1:
            return float(ai_score[0])
        return ai_score.astype(float)

    def get_ai_score_cls(self, features, model_dict: Mapping[str, Any]) -> float | np.ndarray:
        cols = model_dict["features"]
        pipeline = model_dict["pipeline"]
        df = self._build_df(features, cols)
        probs = pipeline.predict_proba(df)
        classes = getattr(pipeline, "classes_", np.arange(probs.shape[1]))
        if len(classes) >= 3:
            idx_down = int(np.argmin(classes))
            idx_up = int(np.argmax(classes))
        else:
            idx_down = 0
            idx_up = min(1, len(classes) - 1)
        prob_down = probs[:, idx_down]
        prob_up = probs[:, idx_up]
        denom = prob_up + prob_down
        scores = np.where(denom == 0, 0.0, (prob_up - prob_down) / denom)
        if scores.size == 1:
            return float(scores[0])
        return scores.astype(float)

    def get_reg_prediction(
        self, features: Mapping[str, float | int | None], model_dict: Mapping[str, Any]
    ) -> float:
        cols = model_dict["features"]
        pipeline = model_dict["pipeline"]
        df = self._build_df(features, cols)
        return float(pipeline.predict(df)[0])

    def get_vol_prediction(
        self, features: Mapping[str, float | int | None], model_dict: Mapping[str, Any]
    ) -> float:
        return self.get_reg_prediction(features, model_dict)

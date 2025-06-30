import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class AIModelPredictor:
    """封装AI模型的加载与推理"""

    def __init__(self, model_paths: dict):
        self.models: dict = {}
        self.calibrators: dict = {}
        base_dir = Path(__file__).resolve().parent
        for period, path_dict in model_paths.items():
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

    def _build_df(self, features: dict, train_cols: list[str]) -> pd.DataFrame:
        row = {c: [features.get(c, np.nan)] for c in train_cols}
        df = pd.DataFrame(row)
        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return df

    def predict_proba(self, period: str, direction: str, features: dict) -> float:
        model = self.models[period][direction]
        df = self._build_df(features, model["features"])
        proba = model["pipeline"].predict_proba(df)[:, 1]
        cal = self.calibrators[period].get(direction)
        if cal is not None:
            proba = cal.transform(proba.reshape(-1, 1)).ravel()
        return float(proba[0])

    def get_ai_score(self, features: dict, model_up: dict, model_down: dict, calibrator_up=None, calibrator_down=None) -> float:
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
        return float(ai_score)

    def get_ai_score_cls(self, features: dict, model_dict: dict) -> float:
        cols = model_dict["features"]
        df = self._build_df(features, cols)
        probs = model_dict["pipeline"].predict_proba(df)[0]
        classes = getattr(model_dict["pipeline"], "classes_", np.arange(len(probs)))
        if len(classes) >= 3:
            idx_down = int(np.argmin(classes))
            idx_up = int(np.argmax(classes))
        else:
            idx_down = 0
            idx_up = min(1, len(probs) - 1)
        prob_down = probs[idx_down]
        prob_up = probs[idx_up]
        denom = prob_up + prob_down
        if denom == 0:
            return 0.0
        return float((prob_up - prob_down) / denom)

    def get_reg_prediction(self, features: dict, model_dict: dict) -> float:
        df = self._build_df(features, model_dict["features"])
        return float(model_dict["pipeline"].predict(df)[0])

    def get_vol_prediction(self, features: dict, model_dict: dict) -> float:
        return self.get_reg_prediction(features, model_dict)

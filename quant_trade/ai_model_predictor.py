from __future__ import annotations

import numpy as np
import pandas as pd
from .utils.soft_clip import soft_clip


class AIModelPredictor:
    """封装 AI 模型推理相关逻辑"""

    def get_ai_score(self, features, model_up, model_down, calibrator_up=None, calibrator_down=None):
        def _build_df(model_dict):
            cols = model_dict["features"]
            row = {c: [features.get(c, np.nan)] for c in cols}
            df = pd.DataFrame(row)
            df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
            return df

        X_up = _build_df(model_up)
        X_down = _build_df(model_down)
        prob_up = model_up["pipeline"].predict_proba(X_up)[:, 1]
        prob_down = model_down["pipeline"].predict_proba(X_down)[:, 1]
        if calibrator_up is not None:
            prob_up = calibrator_up.transform(prob_up.reshape(-1, 1)).ravel()
        if calibrator_down is not None:
            prob_down = calibrator_down.transform(prob_down.reshape(-1, 1)).ravel()
        denom = prob_up + prob_down
        ai_score = np.where(denom == 0, 0.0, (prob_up - prob_down) / denom)
        ai_score = soft_clip(ai_score, k=1.0)
        if ai_score.size == 1:
            return float(ai_score[0])
        return ai_score

    def get_ai_score_cls(self, features, model_dict):
        cols = model_dict["features"]
        row = {c: [features.get(c, np.nan)] for c in cols}
        df = pd.DataFrame(row)
        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)

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
        ai_score = (prob_up - prob_down) / denom
        return float(soft_clip(ai_score, k=1.0))

    def get_vol_prediction(self, features, model_dict):
        lgb_model = model_dict["pipeline"]
        train_cols = model_dict["features"]
        row_data = {col: [features.get(col, 0)] for col in train_cols}
        X_df = pd.DataFrame(row_data)
        X_df = X_df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return float(lgb_model.predict(X_df)[0])

    def get_reg_prediction(self, features, model_dict):
        model = model_dict["pipeline"]
        cols = model_dict["features"]
        row_data = {c: [features.get(c, 0)] for c in cols}
        df = pd.DataFrame(row_data)
        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return float(model.predict(df)[0])

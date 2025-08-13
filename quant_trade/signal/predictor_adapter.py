"""兼容旧接口的预测器包装."""
from __future__ import annotations

from typing import Any

import pandas as pd
from ..ai_model_predictor import AIModelPredictor


class PredictorAdapter:
    """轻量包装 `AIModelPredictor`，保留旧方法名。"""

    def __init__(self, ai_predictor: AIModelPredictor | None) -> None:
        self.ai_predictor = ai_predictor

    def get_ai_score(self, features: Any, *args: Any, **kwargs: Any) -> float:
        if isinstance(features, pd.DataFrame):
            if len(features.index):
                features = features.iloc[0].to_dict()
            else:
                features = {}
        elif isinstance(features, pd.Series):
            features = features.to_dict()
        if self.ai_predictor is None:
            return 0.0
        return self.ai_predictor.get_ai_score(features, *args, **kwargs)

    def get_ai_score_cls(self, features: Any, *args: Any, **kwargs: Any) -> float:
        if isinstance(features, pd.DataFrame):
            if len(features.index):
                features = features.iloc[0].to_dict()
            else:
                features = {}
        elif isinstance(features, pd.Series):
            features = features.to_dict()
        if self.ai_predictor is None:
            return 0.0
        return self.ai_predictor.get_ai_score_cls(features, *args, **kwargs)

    def get_vol_prediction(self, features: Any, *args: Any, **kwargs: Any):
        if isinstance(features, pd.DataFrame):
            if len(features.index):
                features = features.iloc[0].to_dict()
            else:
                features = {}
        elif isinstance(features, pd.Series):
            features = features.to_dict()
        if self.ai_predictor is None:
            return None
        return self.ai_predictor.get_vol_prediction(features, *args, **kwargs)

    def get_reg_prediction(self, features: Any, *args: Any, **kwargs: Any):
        if isinstance(features, pd.DataFrame):
            if len(features.index):
                features = features.iloc[0].to_dict()
            else:
                features = {}
        elif isinstance(features, pd.Series):
            features = features.to_dict()
        if self.ai_predictor is None:
            return None
        return self.ai_predictor.get_reg_prediction(features, *args, **kwargs)

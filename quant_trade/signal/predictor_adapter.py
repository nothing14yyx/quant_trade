from typing import Any
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from quant_trade.logging import get_logger

from ..ai_model_predictor import AIModelPredictor

logger = get_logger(__name__)


class PredictorAdapter:
    """包装 AIModelPredictor，提供便捷预测接口"""

    def __init__(
        self,
        ai_predictor: AIModelPredictor | None,
        rise_transform: str = "none",
        boxcox_lambda: dict | None = None,
    ) -> None:
        self.ai_predictor = ai_predictor
        self.rise_transform = rise_transform
        self.boxcox_lambda = boxcox_lambda or {}

    def get_ai_score(
        self,
        features: Any,
        model_up: dict,
        model_down: dict,
        calibrator_up=None,
        calibrator_down=None,
    ) -> float:
        """根据上涨/下跌模型概率差值计算 AI 得分"""

        if isinstance(features, pd.DataFrame):
            if not len(features.index):
                features = {}
            else:
                features = features.iloc[0].to_dict()
        elif isinstance(features, pd.Series):
            features = features.to_dict()

        if self.ai_predictor is None:
            return 0.0

        return self.ai_predictor.get_ai_score(
            features,
            model_up,
            model_down,
            calibrator_up,
            calibrator_down,
        )

    def get_ai_score_cls(self, features: Any, model_dict: dict) -> float:
        """从单个分类模型计算 AI 得分"""
        if self.ai_predictor is None:
            return 0.0
        return self.ai_predictor.get_ai_score_cls(features, model_dict)

    def get_vol_prediction(self, features: Any, model_dict: dict):
        """根据回归模型预测未来波动率"""
        if self.ai_predictor is None:
            return None
        return self.ai_predictor.get_vol_prediction(features, model_dict)

    def get_reg_prediction(
        self,
        features: Any,
        model_dict: dict,
        tag: str | None = None,
        period: str | None = None,
    ):
        """通用回归模型预测, 根据 tag 应用逆变换"""
        if self.ai_predictor is None:
            return None
        pred = self.ai_predictor.get_reg_prediction(features, model_dict)
        if tag == "rise" and self.rise_transform != "none":
            if self.rise_transform == "log":
                return float(np.expm1(pred))
            if self.rise_transform == "boxcox":
                lmbda = None
                if period is not None:
                    lmbda = self.boxcox_lambda.get(period)
                if lmbda is not None:
                    return float(inv_boxcox(pred, lmbda) - 1.0)
        return pred

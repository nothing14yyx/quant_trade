import numpy as np


class RiskManager:
    """风险控制逻辑"""

    def __init__(self, cap: float = 5.0):
        self.cap = cap

    def fused_to_risk(
        self, fused_score: float, logic_score: float, env_score: float
    ) -> float:
        """根据逻辑得分计算风控分数, 并按 ``cap`` 上限限制."""

        denom = max(abs(logic_score), 1e-6)
        risk = abs(fused_score) / denom
        return float(min(risk, self.cap))

    def calc_risk(
        self,
        env_score: float,
        pred_vol: float | None = None,
        oi_change: float | None = None,
        *,
        quantile: float = 0.75,
    ) -> float:
        """综合环境得分、预测波动率和 OI 变化计算风险值"""

        values = [abs(env_score)]
        if pred_vol is not None:
            values.append(abs(pred_vol))
        if oi_change is not None:
            values.append(abs(oi_change))

        risk = float(np.quantile(values, quantile))
        return min(risk, self.cap)

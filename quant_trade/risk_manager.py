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

    def optimize_weights(
        self, scores: list[float], *, total: float = 1.0, max_weight: float | None = None
    ) -> list[float]:
        """根据多币种得分优化资金权重。

        Args:
            scores: 各币种的信号得分列表。
            total: 权重总和上限，默认 1.0 表示满仓。
            max_weight: 单币种权重上限，``None`` 表示不限制。

        Returns:
            与 ``scores`` 等长的权重列表。
        """

        if not scores:
            return []

        arr = np.abs(np.asarray(scores, dtype=float))
        if arr.sum() == 0:
            return [0.0] * len(scores)

        weights = arr / arr.sum() * total
        if max_weight is not None:
            weights = np.minimum(weights, max_weight)
        return weights.tolist()

import numpy as np


class RiskManager:
    """风险控制逻辑"""

    def __init__(self, cap: float = 5.0):
        self.cap = cap

    def fused_to_risk(self, fused_score: float, logic_score: float, env_score: float) -> float:
        denom = max(abs(logic_score), 1e-6)
        risk = abs(fused_score) / denom
        return float(min(risk, self.cap))

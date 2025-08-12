from enum import StrEnum


class RiskReason(StrEnum):
    """风险过滤或归零原因枚举"""

    NO_DIRECTION = "no_direction"
    VOL_RATIO = "vol_ratio"
    MIN_POS = "min_pos"
    VOTE_FILTER = "vote_filter"
    FUNDING_CONFLICT = "funding_conflict"
    CONFLICT_FILTER = "conflict_filter"
    VOTE_PENALTY = "vote_penalty"
    FUNDING_PENALTY = "funding_penalty"
    CONFLICT_PENALTY = "conflict_penalty"
    RISK_LIMIT = "risk_limit"
    OI_OVERHEAT = "oi_overheat"


# 旧名称的兼容别名
ZeroReason = RiskReason

__all__ = ["RiskReason", "ZeroReason"]

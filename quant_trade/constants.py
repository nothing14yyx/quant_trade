from enum import StrEnum

class ZeroReason(StrEnum):
    """仓位被归零的原因枚举"""
    NO_DIRECTION = "no_direction"
    VOL_RATIO = "vol_ratio"
    MIN_POS = "min_pos"
    VOTE_FILTER = "vote_filter"
    FUNDING_CONFLICT = "funding_conflict"
    CONFLICT_FILTER = "conflict_filter"

__all__ = ["ZeroReason"]

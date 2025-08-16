from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

PERIODS: Tuple[str, str, str] = ("1h", "4h", "d1")


@dataclass
class Vote:
    """简单的投票数据结构。"""

    dir: int
    prob: float

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.dir not in (-1, 0, 1):
            raise ValueError("dir must be -1, 0 or 1")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")


def fuse_votes(
    v1h: Vote,
    v4h: Vote,
    vd1: Vote,
    w: Tuple[float, float, float] = (1.0, 0.2, 0.1),
    veto: bool = True,
) -> int:
    """融合多周期投票并应用否决逻辑。

    Parameters
    ----------
    v1h, v4h, vd1:
        三个周期的投票结果。
    w:
        各周期对应的权重 ``(1h, 4h, d1)``。
    veto:
        是否启用否决逻辑。
    """

    periods = PERIODS
    votes = {periods[0]: v1h, periods[1]: v4h, periods[2]: vd1}
    weighted = sum(votes[p].dir * w[i] for i, p in enumerate(periods))
    fused_dir = 1 if weighted > 0 else -1 if weighted < 0 else 0

    if veto:
        if votes[periods[1]].dir == -votes[periods[0]].dir and abs(votes[periods[1]].prob - votes[periods[0]].prob) > 0.15:
            return 0
        if votes[periods[2]].dir == -votes[periods[0]].dir and votes[periods[2]].prob > 0.60:
            return 0

    if fused_dir == votes[periods[0]].dir and votes[periods[0]].prob > 0.55:
        return fused_dir
    return 0

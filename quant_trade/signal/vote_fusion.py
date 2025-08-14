from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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

    weighted = v1h.dir * w[0] + v4h.dir * w[1] + vd1.dir * w[2]
    fused_dir = 1 if weighted > 0 else -1 if weighted < 0 else 0

    if veto:
        if v4h.dir == -v1h.dir and abs(v4h.prob - v1h.prob) > 0.15:
            return 0
        if vd1.dir == -v1h.dir and vd1.prob > 0.60:
            return 0

    if fused_dir == v1h.dir and v1h.prob > 0.55:
        return fused_dir
    return 0

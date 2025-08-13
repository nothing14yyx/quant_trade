# -*- coding: utf-8 -*-
"""多周期信号融合与相关工具函数。"""

from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np


def consensus_check(s1: float, s2: float, s3: float, min_agree: int = 2) -> int:
    """多周期方向共振检查。

    Parameters
    ----------
    s1, s2, s3 : float
        三个周期的得分。
    min_agree : int, default 2
        至少多少个周期同向才认为有效。

    Returns
    -------
    int
        方向标记, 1 表示向上, -1 表示向下, 0 表示无共识。
    """
    signs = np.sign([s1, s2, s3])
    non_zero = [g for g in signs if g != 0]
    if len(non_zero) < min_agree:
        return 0
    cnt = Counter(non_zero)
    if cnt.most_common(1)[0][1] >= min_agree:
        return int(cnt.most_common(1)[0][0])
    return int(np.sign(np.sum(signs)))


def crowding_protection(
    scores,
    current_score: float,
    base_th: float = 0.2,
    *,
    max_same_direction_rate: float = 0.9,
    equity_drawdown: float = 0.0,
) -> float:
    """根据同向排名抑制过度拥挤的信号, 返回衰减系数。"""
    if not scores or len(scores) < 30:
        return 1.0

    arr = np.array(scores, dtype=float)
    mask = np.abs(arr) >= base_th * 0.8
    arr = arr[mask]
    signs = [s for s in np.sign(arr) if s != 0]
    total = len(signs)
    if total == 0:
        return 1.0
    pos_counts = Counter(signs)
    dominant_dir, cnt = pos_counts.most_common(1)[0]
    if np.sign(current_score) != dominant_dir:
        return 1.0

    ratio = cnt / total
    abs_arr = np.abs(arr)
    rank_pct = float((abs_arr <= abs(current_score)).mean())
    ratio_intensity = max(
        0.0,
        (ratio - max_same_direction_rate) / (1 - max_same_direction_rate),
    )
    rank_intensity = max(0.0, rank_pct - 0.8) / 0.2
    intensity = min(1.0, max(ratio_intensity, rank_intensity))

    factor = 1.0 - 0.2 * intensity
    factor *= max(0.6, 1 - equity_drawdown)
    return factor


def get_ic_weights(core) -> Tuple[float, float, float]:
    """从 ``RobustSignalGenerator`` 的缓存中读取 IC 权重。"""
    cache = getattr(core, "_fuse_cache", None)
    weights = cache.get("ic_weights") if cache is not None else None
    if weights is not None:
        return weights

    ic_scores = getattr(core, "ic_scores", {})
    ic_periods = {
        "1h": ic_scores.get("1h", 1.0),
        "4h": ic_scores.get("4h", 1.0),
        "d1": ic_scores.get("d1", 1.0),
    }
    weights = core.get_ic_period_weights(ic_periods)
    if cache is not None:
        cache.set("ic_weights", weights)
    return weights


def fuse_scores(
    scores: dict,
    ic_weights: Tuple[float, float, float],
    strong_confirm_4h: bool,
    *,
    cycle_weight: dict | None = None,
    conflict_mult: float = 0.7,
) -> tuple[float, bool, bool, bool]:
    """按照多周期共振逻辑融合得分。"""
    s1, s4, sd = scores["1h"], scores["4h"], scores["d1"]
    w1, w4, wd = ic_weights

    consensus_dir = consensus_check(s1, s4, sd)
    consensus_all = (
        consensus_dir != 0 and np.sign(s1) == np.sign(s4) == np.sign(sd)
    )
    consensus_14 = (
        consensus_dir != 0 and np.sign(s1) == np.sign(s4) and not consensus_all
    )
    consensus_4d1 = (
        consensus_dir != 0 and np.sign(s4) == np.sign(sd) and np.sign(s1) != np.sign(s4)
    )

    cw = cycle_weight or {}
    if consensus_all:
        fused = w1 * s1 + w4 * s4 + wd * sd
        conf = 1.1
        if strong_confirm_4h:
            conf *= 1.05
        fused *= cw.get("strong", 1.0)
    elif consensus_14:
        total = w1 + w4
        fused = (w1 / total) * s1 + (w4 / total) * s4
        conf = 0.9
        fused *= cw.get("weak", 1.0)
    elif consensus_4d1:
        total = w4 + wd
        fused = (w4 / total) * s4 + (wd / total) * sd
        conf = 0.9
        fused *= cw.get("weak", 1.0)
    else:
        fused = s1
        conf = 1.0

    fused_score = fused * conf
    if (
        np.sign(s1) != 0
        and (
            (np.sign(s4) != 0 and np.sign(s1) != np.sign(s4))
            or (np.sign(sd) != 0 and np.sign(s1) != np.sign(sd))
        )
    ):
        fused_score *= cw.get("opposite", 1.0)
    if not (consensus_all or consensus_14 or consensus_4d1):
        fused_score *= conflict_mult
    return fused_score, consensus_all, consensus_14, consensus_4d1

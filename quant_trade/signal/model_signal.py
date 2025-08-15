"""Model signal utilities.

This module provides helpers to convert model output probabilities
into standardized trading scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ModelSignalCfg:
    """Configuration for converting model probabilities to scores.

    This configuration supports two modes:

    1. **Continuous**: interpret ``proba`` as a single probability and map
       it to a score using a ``tanh`` transformation controlled by
       ``center``/``scale``/``clip``.
    2. **Categorical**: interpret ``proba`` as an array where indexes
       ``ix_up`` and ``ix_down`` correspond to upward and downward class
       probabilities. ``p_min_up``/``p_min_down`` and ``margin_min`` define
       thresholds for emitting a directional score.
    """

    center: float = 0.5
    scale: float = 5.0
    clip: float | None = 1.0
    p_min_up: float = 0.60
    p_min_down: float = 0.60
    margin_min: float = 0.10
    ix_up: int = 2
    ix_down: int = 0


def model_score_from_proba(
    proba: float | Sequence[float] | np.ndarray,
    cfg: ModelSignalCfg | None = None,
) -> float | None | np.ndarray:
    """Convert model probability to a standardized score.

    Depending on the shape of ``proba`` this function either maps a single
    probability to a continuous score or derives a discrete directional
    score from categorical class probabilities.
    """

    cfg = cfg or ModelSignalCfg()
    p = np.asarray(proba, dtype=float)

    if p.ndim == 0 or p.size == 1:
        score = np.tanh((p - cfg.center) * cfg.scale)
        if cfg.clip is not None:
            score = np.clip(score, -cfg.clip, cfg.clip)
        return score.item()

    p_up = float(p[cfg.ix_up])
    p_down = float(p[cfg.ix_down])
    margin = p_up - p_down
    if p_up >= cfg.p_min_up and margin >= cfg.margin_min:
        return 1.0
    if p_down >= cfg.p_min_down and -margin >= cfg.margin_min:
        return -1.0
    return None


__all__ = ["ModelSignalCfg", "model_score_from_proba"]

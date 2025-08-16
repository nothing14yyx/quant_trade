"""Model signal utilities.

This module provides helpers to convert model output probabilities
into standardized trading scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
       thresholds for emitting a directional score. 可以通过
       ``symbol_thresholds`` 为不同合约设置特定阈值, 其格式为
       ``{symbol: {"p_min_up": x, "p_min_down": y, "margin_min": z}}``。
    """

    center: float = 0.5
    scale: float = 5.0
    clip: float | None = 1.0
    p_min_up: float = 0.60
    p_min_down: float = 0.60
    margin_min: float = 0.10
    ix_up: int = 2
    ix_down: int = 0
    symbol_thresholds: Mapping[str, Mapping[str, float]] | None = None

    @classmethod
    def from_calibration(
        cls, data: Mapping[str, float] | Sequence[float] | None = None
    ) -> "ModelSignalCfg":
        """Create a config from calibration output.

        Parameters
        ----------
        data:
            校准结果, 可以是 ``{"center": x, "scale": y}`` 的映射,
            也可以是 ``(center, scale[, clip])`` 的序列. 缺失的参数将
            使用默认值。
        """

        if data is None:
            return cls()
        if isinstance(data, Mapping):
            allowed = {
                k: float(data[k])
                for k in ("center", "scale", "clip")
                if k in data
            }
            return cls(**allowed)

        seq = list(data)
        center = float(seq[0]) if len(seq) > 0 else cls.center
        scale = float(seq[1]) if len(seq) > 1 else cls.scale
        clip = (
            float(seq[2]) if len(seq) > 2 else cls.clip
        )  # type: ignore[arg-type]
        return cls(center=center, scale=scale, clip=clip)


def model_score_from_proba(
    proba: float | Sequence[float] | np.ndarray,
    cfg: ModelSignalCfg | None = None,
    symbol: str | None = None,
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

    th = {}
    if symbol and cfg.symbol_thresholds:
        th = cfg.symbol_thresholds.get(symbol, {})
    p_min_up = th.get("p_min_up", cfg.p_min_up)
    p_min_down = th.get("p_min_down", cfg.p_min_down)
    margin_min = th.get("margin_min", cfg.margin_min)

    if p_up >= p_min_up and margin >= margin_min:
        return 1.0
    if p_down >= p_min_down and -margin >= margin_min:
        return -1.0
    return None


def fit_center_scale(
    probas: Sequence[float] | np.ndarray,
    future_returns: Sequence[float] | np.ndarray,
    centers: Sequence[float] | None = None,
    scales: Sequence[float] | None = None,
    clip: float | None = 1.0,
) -> dict[str, float | None]:
    """Find ``center`` 与 ``scale`` 使分数与收益相关性最大。

    Parameters
    ----------
    probas:
        历史概率序列。
    future_returns:
        与 ``probas`` 对齐的未来收益序列。
    centers, scales:
        搜索网格, 默认为 ``[0, 1]`` 与 ``[1, 10]`` 之间的均匀网格。
    clip:
        评分裁剪阈值, 返回结果中会原样保存。

    Returns
    -------
    dict
        包含 ``center``、``scale`` 与 ``clip`` 的字典。
    """

    p = np.asarray(probas, dtype=float).reshape(-1)
    r = np.asarray(future_returns, dtype=float).reshape(-1)
    if p.size == 0 or r.size == 0 or p.size != r.size:
        return {"center": 0.5, "scale": 1.0, "clip": clip}

    if centers is None:
        centers = np.linspace(0.0, 1.0, 21)
    if scales is None:
        scales = np.linspace(1.0, 10.0, 10)

    best_corr = -1.0
    best_c = float(centers[0])
    best_s = float(scales[0])
    for c in centers:
        for s in scales:
            score = np.tanh((p - c) * s)
            if clip is not None:
                score = np.clip(score, -clip, clip)
            corr = np.corrcoef(score, r)[0, 1]
            if np.isnan(corr):
                continue
            abs_corr = abs(corr)
            if abs_corr > best_corr:
                best_corr = abs_corr
                best_c = float(c)
                best_s = float(s if corr >= 0 else -s)

    return {"center": best_c, "scale": best_s, "clip": clip}


__all__ = ["ModelSignalCfg", "model_score_from_proba", "fit_center_scale"]

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
    """Configuration for converting probabilities to scores.

    Attributes
    ----------
    center:
        Probability regarded as a neutral point. Values above imply a
        positive expectation while below imply negative.
    scale:
        Scaling factor controlling how sharply probabilities are mapped to
        scores.
    clip:
        Optional absolute clip applied to the resulting score. ``None``
        disables clipping.
    """

    center: float = 0.5
    scale: float = 5.0
    clip: float | None = 1.0


def model_score_from_proba(
    proba: float | Sequence[float] | np.ndarray,
    cfg: ModelSignalCfg | None = None,
) -> float | np.ndarray:
    """Convert model probability to a standardized score.

    Parameters
    ----------
    proba:
        Probability or an array of probabilities representing the likelihood
        of a positive outcome from the model.
    cfg:
        Optional :class:`ModelSignalCfg` instance providing conversion
        parameters. If omitted a default configuration is used.

    Returns
    -------
    float or :class:`numpy.ndarray`
        Converted score(s) in the range ``[-clip, clip]`` if ``clip`` is set,
        otherwise in ``(-inf, inf)``. Single input probabilities yield a
        float; otherwise a NumPy array is returned.
    """

    cfg = cfg or ModelSignalCfg()
    p = np.asarray(proba, dtype=float)
    score = np.tanh((p - cfg.center) * cfg.scale)
    if cfg.clip is not None:
        score = np.clip(score, -cfg.clip, cfg.clip)
    return score.item() if score.ndim == 0 else score


__all__ = ["ModelSignalCfg", "model_score_from_proba"]

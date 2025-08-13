# -*- coding: utf-8 -*-
"""Dynamic thresholding utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from .utils import smooth_series, _calc_history_base


@dataclass
class SignalThresholdParams:
    """Simplified parameters for signal thresholding."""

    base_th: float = 0.0
    low_base: float = 0.0
    quantile: float = 0.8
    rev_boost: float = 0.0
    rev_th_mult: float = 1.0


@dataclass
class DynamicThresholdParams:
    """Parameters controlling dynamic threshold adjustments."""

    atr_mult: float = 3.63636363636
    atr_cap: float = 0.2
    funding_mult: float = 2.7586206897
    funding_cap: float = 0.2
    adx_div: float = 25.0
    adx_cap: float = 0.2


def compute_dynamic_threshold(history_scores: Iterable[float], window: int, quantile: float):
    """Compute absolute score threshold from history."""
    if not history_scores:
        return None
    arr = np.abs(np.asarray(list(history_scores)[-window:], dtype=float))
    if arr.size == 0:
        return None
    return float(np.quantile(arr, quantile))


@dataclass
class DynamicThresholdInput:
    """Container for metrics used in dynamic threshold calculation."""

    atr: float
    adx: float
    bb_width_chg: float | None = None
    channel_pos: float | None = None
    funding: float = 0.0
    atr_4h: float | None = None
    adx_4h: float | None = None
    atr_d1: float | None = None
    adx_d1: float | None = None
    pred_vol: float | None = None
    pred_vol_4h: float | None = None
    pred_vol_d1: float | None = None
    vix_proxy: float | None = None
    regime: str | None = None
    reversal: bool = False
    base: float | None = None
    low_base: float | None = None
    history_scores: Iterable[float] | None = None
    signal_params: "SignalThresholdParams | None" = None
    dynamic_params: "DynamicThresholdParams | None" = None
    smooth_window: int = 20
    smooth_alpha: float = 0.2
    smooth_limit: float | None = 1.0
    th_window: int = 60
    th_decay: float = 2.0


def _classify_regime(adx: float | None, bb_width_chg: float | None) -> str:
    """Simple market regime classification used when regime is not provided."""
    if adx is None or bb_width_chg is None:
        return "unknown"
    try:
        adx = float(adx)
        bb_chg = float(bb_width_chg)
    except (TypeError, ValueError):
        return "unknown"
    if adx >= 25 and bb_chg > 0:
        return "trend"
    if adx <= 20 and bb_chg < 0:
        return "range"
    return "unknown"


def calc_dynamic_threshold(params: DynamicThresholdInput) -> tuple[float, float]:
    """Calculate dynamic threshold and reversal boost.

    Parameters are provided via :class:`DynamicThresholdInput`.
    """
    sig_p = params.signal_params or SignalThresholdParams()
    dyn_p = params.dynamic_params or DynamicThresholdParams()

    base = sig_p.base_th if params.base is None else params.base
    low_base = sig_p.low_base if params.low_base is None else params.low_base

    if params.history_scores is not None:
        scores = smooth_series(
            params.history_scores,
            window=params.smooth_window,
            alpha=params.smooth_alpha,
        )
        if params.smooth_limit is not None:
            scores = np.clip(scores, -params.smooth_limit, params.smooth_limit)
        hist_base = _calc_history_base(
            scores,
            base,
            sig_p.quantile,
            params.th_window,
            params.th_decay,
            0.12,
        )
    else:
        hist_base = base

    th = hist_base
    atr_eff = abs(params.atr)
    if params.atr_4h is not None:
        atr_eff += 0.5 * abs(params.atr_4h)
    if params.atr_d1 is not None:
        atr_eff += 0.25 * abs(params.atr_d1)
    th += min(dyn_p.atr_cap, atr_eff * dyn_p.atr_mult)

    fund_eff = abs(params.funding)
    if params.pred_vol is not None:
        fund_eff += 0.5 * abs(params.pred_vol)
    if params.pred_vol_4h is not None:
        fund_eff += 0.25 * abs(params.pred_vol_4h)
    if params.pred_vol_d1 is not None:
        fund_eff += 0.15 * abs(params.pred_vol_d1)
    if params.vix_proxy is not None:
        fund_eff += 0.25 * abs(params.vix_proxy)
    th += min(dyn_p.funding_cap, fund_eff * dyn_p.funding_mult)

    adx_eff = abs(params.adx)
    if params.adx_4h is not None:
        adx_eff += 0.5 * abs(params.adx_4h)
    if params.adx_d1 is not None:
        adx_eff += 0.25 * abs(params.adx_d1)
    th += min(dyn_p.adx_cap, adx_eff / dyn_p.adx_div)

    if atr_eff == 0 and adx_eff == 0 and fund_eff == 0:
        th = min(th, hist_base)

    regime = params.regime or _classify_regime(params.adx, params.bb_width_chg)
    if params.reversal:
        th *= sig_p.rev_th_mult

    rev_boost = sig_p.rev_boost
    if regime == "trend":
        th *= 1.05
        rev_boost *= 0.8
    elif regime == "range":
        th *= 0.95
        rev_boost *= 1.2

    return max(th, low_base), rev_boost


class ThresholdingDynamic:
    """Dynamic thresholding helper bound to a generator instance."""

    def __init__(self, owner):
        self.owner = owner

    @staticmethod
    def adaptive_rsi_threshold(
        rsi_series, vol_series, k: float | None = None, window: int = 14
    ):
        """Weighted standard deviation based RSI thresholds."""
        if rsi_series is None or vol_series is None:
            return 30.0, 70.0
        rsi_series = pd.Series(rsi_series).astype(float)
        vol_series = pd.Series(vol_series).astype(float)
        df = pd.DataFrame({"rsi": rsi_series, "vol": vol_series}).dropna()
        if df.empty or len(df) < window:
            return 30.0, 70.0
        roll = df.tail(window)
        weights = roll["vol"].to_numpy()
        rsi_vals = roll["rsi"].to_numpy()
        if not np.isfinite(weights).all() or np.allclose(weights, 0):
            mean = rsi_vals.mean()
            std = rsi_vals.std(ddof=0)
        else:
            mean = np.average(rsi_vals, weights=weights)
            var = np.average((rsi_vals - mean) ** 2, weights=weights)
            std = float(np.sqrt(var))
        if k is None:
            from .core import DEFAULT_RSI_K  # local import to avoid cycle

            k = DEFAULT_RSI_K
        lower = float(mean - k * std)
        upper = float(mean + k * std)
        return lower, upper

    def base(
        self,
        atr,
        adx,
        funding=0,
        *,
        atr_4h=None,
        adx_4h=None,
        atr_d1=None,
        adx_d1=None,
        bb_width_chg=None,
        channel_pos=None,
        pred_vol=None,
        pred_vol_4h=None,
        pred_vol_d1=None,
        vix_proxy=None,
        base=None,
        regime=None,
        low_base=None,
        reversal=False,
        history_scores=None,
    ):
        """Compute dynamic threshold and reversal boost."""
        data = DynamicThresholdInput(
            atr=atr,
            adx=adx,
            bb_width_chg=bb_width_chg,
            channel_pos=channel_pos,
            funding=funding,
            atr_4h=atr_4h,
            adx_4h=adx_4h,
            atr_d1=atr_d1,
            adx_d1=adx_d1,
            pred_vol=pred_vol,
            pred_vol_4h=pred_vol_4h,
            pred_vol_d1=pred_vol_d1,
            vix_proxy=vix_proxy,
            regime=regime,
            reversal=reversal,
            base=base,
            low_base=low_base,
            history_scores=history_scores,
            signal_params=self.owner.signal_params,
            dynamic_params=self.owner.dynamic_th_params,
            smooth_window=getattr(self.owner, "smooth_window", 20),
            smooth_alpha=getattr(self.owner, "smooth_alpha", 0.2),
            smooth_limit=getattr(self.owner, "smooth_limit", 1.0),
            th_window=getattr(self.owner, "th_window", 60),
            th_decay=getattr(self.owner, "th_decay", 2.0),
        )
        return calc_dynamic_threshold(data)


# Re-export for backward compatibility
adaptive_rsi_threshold = ThresholdingDynamic.adaptive_rsi_threshold

__all__ = [
    "compute_dynamic_threshold",
    "DynamicThresholdInput",
    "calc_dynamic_threshold",
    "ThresholdingDynamic",
    "adaptive_rsi_threshold",
]

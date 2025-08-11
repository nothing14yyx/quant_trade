# -*- coding: utf-8 -*-
"""Dynamic thresholding utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable

from .utils import smooth_series, _calc_history_base


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
        params = self.owner.signal_params
        dyn_p = self.owner.dynamic_th_params
        base = params.base_th if base is None else base
        low_base = params.low_base if low_base is None else low_base

        if history_scores is not None:
            scores = smooth_series(
                history_scores,
                window=getattr(self.owner, "smooth_window", 20),
                alpha=getattr(self.owner, "smooth_alpha", 0.2),
            )
            limit = getattr(self.owner, "smooth_limit", 1.0)
            if limit is not None:
                scores = np.clip(scores, -limit, limit)
            hist_base = _calc_history_base(
                scores,
                base,
                params.quantile,
                getattr(self.owner, "th_window", 60),
                getattr(self.owner, "th_decay", 2.0),
                0.12,
            )
        else:
            hist_base = base

        th = hist_base
        atr_eff = abs(atr)
        if atr_4h is not None:
            atr_eff += 0.5 * abs(atr_4h)
        if atr_d1 is not None:
            atr_eff += 0.25 * abs(atr_d1)
        th += min(dyn_p.atr_cap, atr_eff * dyn_p.atr_mult)

        fund_eff = abs(funding)
        if pred_vol is not None:
            fund_eff += 0.5 * abs(pred_vol)
        if pred_vol_4h is not None:
            fund_eff += 0.25 * abs(pred_vol_4h)
        if pred_vol_d1 is not None:
            fund_eff += 0.15 * abs(pred_vol_d1)
        if vix_proxy is not None:
            fund_eff += 0.25 * abs(vix_proxy)
        th += min(dyn_p.funding_cap, fund_eff * dyn_p.funding_mult)

        adx_eff = abs(adx)
        if adx_4h is not None:
            adx_eff += 0.5 * abs(adx_4h)
        if adx_d1 is not None:
            adx_eff += 0.25 * abs(adx_d1)
        th += min(dyn_p.adx_cap, adx_eff / dyn_p.adx_div)

        if atr_eff == 0 and adx_eff == 0 and fund_eff == 0:
            th = min(th, hist_base)

        if regime is None:
            regime = self.owner.classify_regime(adx, bb_width_chg, channel_pos)

        if reversal:
            th *= params.rev_th_mult

        rev_boost = params.rev_boost
        if regime == "trend":
            th *= 1.05
            rev_boost *= 0.8
        elif regime == "range":
            th *= 0.95
            rev_boost *= 1.2

        return max(th, low_base), rev_boost

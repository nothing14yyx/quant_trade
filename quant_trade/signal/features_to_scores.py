"""Utilities to convert raw features into factor scores.

This module extracts the scoring logic from :class:`FactorScorerImpl`
so that functions can be used independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from quant_trade.logging import get_logger
from .utils import adjust_score, volume_guard

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions migrated from ``core.py``
# ---------------------------------------------------------------------------

def record_volume_ratio(core, ratio: float | None, symbol: str | None = None) -> None:
    """Record ``ratio`` into volume history for ``symbol``."""
    if ratio is None:
        return
    cache = core._get_symbol_cache(symbol)
    cache.setdefault(
        "volume_ratio_history", core.volume_ratio_history.__class__(maxlen=core.volume_ratio_history.maxlen)
    ).append(float(ratio))


def get_volume_ratio_thresholds(core, symbol: str | None = None) -> tuple[float, float]:
    """Return dynamic volume ratio thresholds for ``symbol``."""
    cache = core._get_symbol_cache(symbol)
    hist = cache.get("volume_ratio_history")
    if not hist or len(hist) < 10:
        return (
            core.volume_guard_params["ratio_low"],
            core.volume_guard_params["ratio_high"],
        )
    arr = np.asarray(hist, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 10:
        return (
            core.volume_guard_params["ratio_low"],
            core.volume_guard_params["ratio_high"],
        )
    low = float(np.quantile(arr, core.volume_quantile_low))
    high = float(np.quantile(arr, core.volume_quantile_high))
    return low, high


def ma_cross_logic(core, features: dict, sma_20_1h_prev=None) -> float:
    """Compute multiplier based on MA5/MA20 cross logic."""
    sma5 = core.get_feat_value(features, "sma_5_1h", None)
    sma20 = core.get_feat_value(features, "sma_20_1h", None)
    ma_ratio = core.get_feat_value(features, "ma_ratio_5_20", 1.0)
    if sma5 is None or sma20 is None:
        return 1.0
    slope = 0.0
    if sma_20_1h_prev not in (None, 0):
        slope = (sma20 - sma_20_1h_prev) / sma_20_1h_prev
    if (ma_ratio > 1.02 and slope > 0) or (ma_ratio < 0.98 and slope < 0):
        return 1.15
    if (ma_ratio > 1.02 and slope < 0) or (ma_ratio < 0.98 and slope > 0):
        return 0.85
    return 1.0


# ---------------------------------------------------------------------------
# Core scoring utilities copied from ``FactorScorerImpl``
# ---------------------------------------------------------------------------

def _extract_feature_arrays(core, rows, period: str) -> dict[str, np.ndarray]:
    """将输入的多行特征提取为按列存储的 ``ndarray``。

    ``rows`` 可以是 ``list[Mapping]`` 或带字段名的 ``ndarray``。"""

    names_defaults = {
        f"price_vs_ma200_{period}": 0.0,
        f"ema_slope_50_{period}": 0.0,
        f"adx_{period}": 0.0,
        f"rsi_{period}": 50.0,
        f"macd_hist_{period}": 0.0,
        f"atr_pct_{period}": 0.0,
        f"bb_width_{period}": 0.0,
        f"vol_ma_ratio_{period}": 0.0,
        f"buy_sell_ratio_{period}": 1.0,
        f"funding_rate_{period}": 0.0,
        f"funding_rate_anom_{period}": 0.0,
        f"channel_pos_{period}": 0.5,
    }

    n = len(rows)
    out: dict[str, np.ndarray] = {}
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        for k, d in names_defaults.items():
            if k in rows.dtype.names:
                out[k] = rows[k].astype(float)
            else:
                out[k] = np.full(n, d, dtype=float)
    else:
        for k, d in names_defaults.items():
            out[k] = np.array(
                [core.get_feat_value(row, k, d) for row in rows], dtype=float
            )
    return out


def _score_vectorized(core, rows, period: str) -> dict[str, np.ndarray]:
    """矢量化计算多行 ``rows`` 的因子得分。"""

    arrs = _extract_feature_arrays(core, rows, period)

    trend_raw = (
        np.tanh(arrs[f"price_vs_ma200_{period}"] * 5)
        + np.tanh(arrs[f"ema_slope_50_{period}"] * 5)
        + 0.5 * np.tanh(arrs[f"adx_{period}"] / 50)
    )

    momentum_raw = (
        (arrs[f"rsi_{period}"] - 50) / 50
        + np.tanh(arrs[f"macd_hist_{period}"] * 5)
    )

    volatility_raw = (
        np.tanh(arrs[f"atr_pct_{period}"] * 8)
        + np.tanh(arrs[f"bb_width_{period}"] * 2)
    )

    volume_raw = (
        np.tanh(arrs[f"vol_ma_ratio_{period}"])
        + np.tanh((arrs[f"buy_sell_ratio_{period}"] - 1) * 2)
    )

    sentiment_raw = np.tanh(arrs[f"funding_rate_{period}"] * 4000)

    f_rate = arrs[f"funding_rate_{period}"]
    f_anom = arrs[f"funding_rate_anom_{period}"]
    thr = 0.0005
    funding_raw = np.where(
        np.abs(f_rate) > thr, -np.tanh(f_rate * 4000), np.tanh(f_rate * 4000)
    )
    funding_raw = np.where(np.abs(f_rate) < 0.001, 0.0, funding_raw)
    funding_raw = funding_raw + np.tanh(f_anom * 50)

    scores = {
        "trend": np.tanh(trend_raw),
        "momentum": np.tanh(momentum_raw),
        "volatility": np.tanh(volatility_raw),
        "volume": np.tanh(volume_raw),
        "sentiment": np.tanh(sentiment_raw),
        "funding": np.tanh(funding_raw),
    }

    pos = arrs[f"channel_pos_{period}"]
    for k, v in scores.items():
        v = np.where((pos > 1) & (v > 0), v * 1.2, v)
        v = np.where((pos < 0) & (v < 0), v * 1.2, v)
        v = np.where((pos > 0.9) & (v > 0), v * 0.8, v)
        v = np.where((pos < 0.1) & (v < 0), v * 0.8, v)
        scores[k] = v
    return scores


def score(core, features: Mapping[str, Any], period: str) -> dict[str, float]:
    """Compute raw factor scores for单条特征."""

    return get_factor_scores_batch(core, [features], period)[0]


def get_factor_scores_batch(core, features_iter, period: str):
    """批量计算因子分。``features_iter`` 可以是 ``list[Mapping]`` 或 ``ndarray``。"""

    if isinstance(features_iter, np.ndarray):
        rows = list(features_iter)
    else:
        rows = list(features_iter)

    results: list[dict[str, float] | None] = []
    to_compute_idx: list[int] = []
    to_compute_rows: list[Any] = []
    for idx, row in enumerate(rows):
        if isinstance(row, np.void):  # structured array row
            mapping = {name: row[name] for name in row.dtype.names}
        else:
            mapping = dict(row)
        key = core._make_cache_key(mapping, period)
        cached = core._factor_cache.get(key)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)
            to_compute_idx.append(idx)
            to_compute_rows.append(mapping)

    if to_compute_rows:
        arrays_scores = _score_vectorized(core, to_compute_rows, period)
        n = len(to_compute_rows)
        for j in range(n):
            sc = {k: float(v[j]) for k, v in arrays_scores.items()}
            mapping = to_compute_rows[j]
            key = core._make_cache_key(mapping, period)
            core._factor_cache.set(key, sc)
            results[to_compute_idx[j]] = sc

    return results


def calc_factor_scores(core, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
    w1 = weights.copy()
    w4 = weights.copy()
    for k in ("trend", "momentum", "volume"):
        w1[k] = w1.get(k, 0) * 0.7
        w4[k] = w4.get(k, 0) * 0.7
    scores = {
        "1h": core.combine_score(ai_scores["1h"], factor_scores["1h"], w1),
        "4h": core.combine_score(ai_scores["4h"], factor_scores["4h"], w4),
        "d1": core.combine_score(ai_scores["d1"], factor_scores["d1"], weights),
    }
    logger.debug("factor scores: %s", scores)
    return scores


def calc_factor_scores_vectorized(core, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
    w1 = weights.copy()
    w4 = weights.copy()
    for k in ("trend", "momentum", "volume"):
        w1[k] = w1.get(k, 0) * 0.7
        w4[k] = w4.get(k, 0) * 0.7
    return {
        "1h": core.combine_score_vectorized(ai_scores["1h"], factor_scores["1h"], w1),
        "4h": core.combine_score_vectorized(ai_scores["4h"], factor_scores["4h"], w4),
        "d1": core.combine_score_vectorized(ai_scores["d1"], factor_scores["d1"], weights),
    }


def apply_local_adjustments(
    core,
    scores: dict,
    raw_feats: dict,
    factor_scores: dict,
    deltas: dict,
    rise_pred_1h: float | None = None,
    drawdown_pred_1h: float | None = None,
    symbol: str | None = None,
) -> tuple[dict, dict]:
    adjusted = scores.copy()
    details: dict[str, Any] = {}

    for p in adjusted:
        adjusted[p] = core._apply_delta_boost(adjusted[p], deltas.get(p, {}))

    prev_ma20 = raw_feats["1h"].get("sma_20_1h_prev")
    ma_coeff = ma_cross_logic(core, raw_feats["1h"], prev_ma20)
    adjusted["1h"] *= ma_coeff
    details["ma_cross"] = int(np.sign(ma_coeff - 1.0))

    if rise_pred_1h is not None and drawdown_pred_1h is not None:
        delta = rise_pred_1h - abs(drawdown_pred_1h)
        if delta >= 0.01:
            adj = np.tanh(delta * 5) * 0.5
            adjusted["1h"] *= 1 + adj
            details["rise_drawdown_adj"] = adj
        else:
            details["rise_drawdown_adj"] = 0.0

    strong_confirm_4h = (
        factor_scores["4h"]["trend"] > 0
        and factor_scores["4h"]["momentum"] > 0
        and factor_scores["4h"]["volatility"] > 0
        and adjusted["4h"] > 0
    ) or (
        factor_scores["4h"]["trend"] < 0
        and factor_scores["4h"]["momentum"] < 0
        and factor_scores["4h"]["volatility"] < 0
        and adjusted["4h"] < 0
    )
    details["strong_confirm_4h"] = strong_confirm_4h

    macd_diff = raw_feats["1h"].get("macd_hist_diff_1h_4h")
    rsi_diff = raw_feats["1h"].get("rsi_diff_1h_4h")
    if (
        macd_diff is not None
        and rsi_diff is not None
        and macd_diff < 0
        and rsi_diff < -8
    ):
        if strong_confirm_4h:
            logger.debug(
                "momentum misalign macd_diff=%.3f rsi_diff=%.3f -> strong_confirm=False",
                macd_diff,
                rsi_diff,
            )
        strong_confirm_4h = False
        details["strong_confirm_4h"] = False

    if (
        macd_diff is not None
        and rsi_diff is not None
        and abs(macd_diff) < 5
        and abs(rsi_diff) < 15
    ):
        strong_confirm_4h = True
        details["strong_confirm_4h"] = True

    for p in ["1h", "4h", "d1"]:
        sent = factor_scores[p]["sentiment"]
        before = adjusted[p]
        adjusted[p] = adjust_score(
            adjusted[p],
            sent,
            core.sentiment_alpha,
            cap_scale=core.cap_positive_scale,
        )
        if before != adjusted[p]:
            logger.debug(
                "sentiment %.2f adjust %s: %.3f -> %.3f",
                sent,
                p,
                before,
                adjusted[p],
            )

    params = core.volume_guard_params.copy()
    q_low, q_high = get_volume_ratio_thresholds(core, symbol)
    params["ratio_low"] = q_low
    params["ratio_high"] = q_high
    r1 = raw_feats["1h"].get("vol_ma_ratio_1h")
    roc1 = raw_feats["1h"].get("vol_roc_1h")
    before = adjusted["1h"]
    adjusted["1h"] = volume_guard(adjusted["1h"], r1, roc1, **params)
    if before != adjusted["1h"]:
        logger.debug(
            "volume guard 1h ratio=%.3f roc=%.3f -> %.3f",
            r1,
            roc1,
            adjusted["1h"],
        )
    if raw_feats.get("4h") is not None:
        r4 = raw_feats["4h"].get("vol_ma_ratio_4h")
        roc4 = raw_feats["4h"].get("vol_roc_4h")
        before4 = adjusted["4h"]
        adjusted["4h"] = volume_guard(adjusted["4h"], r4, roc4, **params)
        if before4 != adjusted["4h"]:
            logger.debug(
                "volume guard 4h ratio=%.3f roc=%.3f -> %.3f",
                r4,
                roc4,
                adjusted["4h"],
            )
    r_d1 = raw_feats["d1"].get("vol_ma_ratio_d1")
    roc_d1 = raw_feats["d1"].get("vol_roc_d1")
    before_d1 = adjusted["d1"]
    adjusted["d1"] = volume_guard(adjusted["d1"], r_d1, roc_d1, **params)
    if before_d1 != adjusted["d1"]:
        logger.debug(
            "volume guard d1 ratio=%.3f roc=%.3f -> %.3f",
            r_d1,
            roc_d1,
            adjusted["d1"],
        )

    for p in ["1h", "4h", "d1"]:
        bs = raw_feats[p].get(f"break_support_{p}")
        br = raw_feats[p].get(f"break_resistance_{p}")
        before_sr = adjusted[p]
        if br:
            adjusted[p] *= 1.1 if adjusted[p] > 0 else 0.8
        if bs:
            adjusted[p] *= 1.1 if adjusted[p] < 0 else 0.8
        if before_sr != adjusted[p]:
            logger.debug(
                "break SR %s bs=%s br=%s %.3f->%.3f",
                p,
                bs,
                br,
                before_sr,
                adjusted[p],
            )
            details[f"break_sr_{p}"] = adjusted[p] - before_sr

    for p in ["1h", "4h", "d1"]:
        perc = raw_feats[p].get(f"boll_perc_{p}")
        vol_ratio = raw_feats[p].get(f"vol_ma_ratio_{p}")
        before_bb = adjusted[p]
        if (
            perc is not None
            and vol_ratio is not None
            and vol_ratio > 1.5
            and (perc >= 0.98 or perc <= 0.02)
        ):
            if perc >= 0.98:
                adjusted[p] *= 1.1 if adjusted[p] > 0 else 0.9
            else:
                adjusted[p] *= 1.1 if adjusted[p] < 0 else 0.9
        if before_bb != adjusted[p]:
            logger.debug(
                "boll breakout %s perc=%.3f vol_ratio=%.3f %.3f->%.3f",
                p,
                perc,
                vol_ratio,
                before_bb,
                adjusted[p],
            )
            details[f"boll_breakout_{p}"] = adjusted[p] - before_bb

    return adjusted, details


# Public API ---------------------------------------------------------------

def get_factor_scores(core, features: Mapping[str, Any], period: str) -> dict[str, float]:
    """Public entry mirroring old interface."""
    return score(core, features, period)

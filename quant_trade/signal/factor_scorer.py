from collections.abc import Mapping
import numpy as np

from .utils import adjust_score, volume_guard
from quant_trade.logging import get_logger

logger = get_logger(__name__)


class FactorScorerImpl:
    """封装因子得分相关逻辑的实现类"""

    def __init__(self, core):
        """参数
        ------
        core : RobustSignalGenerator
            主体对象实例, 用于访问共享的方法与属性
        """
        self.core = core

    # 原 get_factor_scores -> score
    def score(self, features: Mapping, period: str) -> dict:
        key = self.core._make_cache_key(features, period)
        cached = self.core._factor_cache.get(key)
        if cached is not None:
            return cached

        dedup_row = {k: v for k, v in features.items()}
        safe = lambda k, d=0: self.core.get_feat_value(dedup_row, k, d)

        trend_raw = (
            np.tanh(safe(f"price_vs_ma200_{period}", 0) * 5)
            + np.tanh(safe(f"ema_slope_50_{period}", 0) * 5)
            + 0.5 * np.tanh(safe(f"adx_{period}", 0) / 50)
        )

        momentum_raw = (
            (safe(f"rsi_{period}", 50) - 50) / 50
            + np.tanh(safe(f"macd_hist_{period}", 0) * 5)
        )

        volatility_raw = (
            np.tanh(safe(f"atr_pct_{period}", 0) * 8)
            + np.tanh(safe(f"bb_width_{period}", 0) * 2)
        )

        volume_raw = (
            np.tanh(safe(f"vol_ma_ratio_{period}", 0))
            + np.tanh((safe(f"buy_sell_ratio_{period}", 1) - 1) * 2)
        )

        sentiment_raw = np.tanh(safe(f"funding_rate_{period}", 0) * 4000)

        f_rate = safe(f"funding_rate_{period}", 0)
        f_anom = safe(f"funding_rate_anom_{period}", 0)
        thr = 0.0005
        if abs(f_rate) > thr:
            funding_raw = -np.tanh(f_rate * 4000)
        else:
            funding_raw = np.tanh(f_rate * 4000)
        if abs(f_rate) < 0.001:
            funding_raw = 0.0
        funding_raw += np.tanh(f_anom * 50)

        scores = {
            "trend": np.tanh(trend_raw),
            "momentum": np.tanh(momentum_raw),
            "volatility": np.tanh(volatility_raw),
            "volume": np.tanh(volume_raw),
            "sentiment": np.tanh(sentiment_raw),
            "funding": np.tanh(funding_raw),
        }

        pos = safe(f"channel_pos_{period}", 0.5)
        for k, v in scores.items():
            if pos > 1 and v > 0:
                scores[k] = v * 1.2
            elif pos < 0 and v < 0:
                scores[k] = v * 1.2
            elif pos > 0.9 and v > 0:
                scores[k] = v * 0.8
            elif pos < 0.1 and v < 0:
                scores[k] = v * 0.8

        self.core._factor_cache.set(key, scores)
        return scores

    # 原 calc_factor_scores
    def calc_factor_scores(self, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
        w1 = weights.copy()
        w4 = weights.copy()
        for k in ("trend", "momentum", "volume"):
            w1[k] = w1.get(k, 0) * 0.7
            w4[k] = w4.get(k, 0) * 0.7
        scores = {
            "1h": self.core.combine_score(ai_scores["1h"], factor_scores["1h"], w1),
            "4h": self.core.combine_score(ai_scores["4h"], factor_scores["4h"], w4),
            "d1": self.core.combine_score(ai_scores["d1"], factor_scores["d1"], weights),
        }
        logger.debug("factor scores: %s", scores)
        return scores

    # 原 calc_factor_scores_vectorized
    def calc_factor_scores_vectorized(self, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
        w1 = weights.copy()
        w4 = weights.copy()
        for k in ("trend", "momentum", "volume"):
            w1[k] = w1.get(k, 0) * 0.7
            w4[k] = w4.get(k, 0) * 0.7
        return {
            "1h": self.core.combine_score_vectorized(ai_scores["1h"], factor_scores["1h"], w1),
            "4h": self.core.combine_score_vectorized(ai_scores["4h"], factor_scores["4h"], w4),
            "d1": self.core.combine_score_vectorized(ai_scores["d1"], factor_scores["d1"], weights),
        }

    # 原 apply_local_adjustments
    def apply_local_adjustments(
        self,
        scores: dict,
        raw_feats: dict,
        factor_scores: dict,
        deltas: dict,
        rise_pred_1h: float | None = None,
        drawdown_pred_1h: float | None = None,
        symbol: str | None = None,
    ) -> tuple[dict, dict]:
        adjusted = scores.copy()
        details = {}

        for p in adjusted:
            adjusted[p] = self.core._apply_delta_boost(adjusted[p], deltas.get(p, {}))

        prev_ma20 = raw_feats["1h"].get("sma_20_1h_prev")
        ma_coeff = self.core.ma_cross_logic(raw_feats["1h"], prev_ma20)
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
                self.core.sentiment_alpha,
                cap_scale=self.core.cap_positive_scale,
            )
            if before != adjusted[p]:
                logger.debug(
                    "sentiment %.2f adjust %s: %.3f -> %.3f",
                    sent,
                    p,
                    before,
                    adjusted[p],
                )

        params = self.core.volume_guard_params.copy()
        q_low, q_high = self.core.get_volume_ratio_thresholds(symbol)
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

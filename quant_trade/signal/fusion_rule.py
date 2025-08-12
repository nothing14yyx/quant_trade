# -*- coding: utf-8 -*-
"""Rule-based fusion utilities extracted from RobustSignalGenerator."""

from __future__ import annotations

from collections import Counter

import numpy as np
from quant_trade.logging import get_logger

logger = get_logger(__name__)


class FusionRuleBased:
    """封装信号融合与拥挤度保护等规则逻辑。"""

    def __init__(self, core) -> None:
        """Parameters
        ----------
        core : RobustSignalGenerator
            引用核心对象以访问其配置与辅助方法。
        """
        self.core = core

    # ------------------------------------------------------------------
    # 评分融合
    # ------------------------------------------------------------------
    @staticmethod
    def combine_score(ai_score, factor_scores, weights):
        """合并 AI 分数与因子得分。"""
        fused_score = (
            ai_score * weights['ai']
            + factor_scores['trend'] * weights['trend']
            + factor_scores['momentum'] * weights['momentum']
            + factor_scores['volatility'] * weights['volatility']
            + factor_scores['volume'] * weights['volume']
            + factor_scores['sentiment'] * weights['sentiment']
            + factor_scores['funding'] * weights['funding']
        )
        return float(fused_score)

    @staticmethod
    def combine_score_vectorized(ai_scores, factor_scores, weights):
        """向量化计算多个样本的合并得分。"""
        weight_arr = np.array(
            [
                weights['ai'],
                weights['trend'],
                weights['momentum'],
                weights['volatility'],
                weights['volume'],
                weights['sentiment'],
                weights['funding'],
            ],
            dtype=float,
        )
        fs_matrix = np.vstack(
            [
                ai_scores,
                factor_scores['trend'],
                factor_scores['momentum'],
                factor_scores['volatility'],
                factor_scores['volume'],
                factor_scores['sentiment'],
                factor_scores['funding'],
            ]
        )
        return (fs_matrix.T * weight_arr).sum(axis=1).astype(float)

    # ------------------------------------------------------------------
    # 共振与拥挤度保护
    # ------------------------------------------------------------------
    def consensus_check(self, s1, s2, s3, min_agree: int = 2):
        """多周期方向共振检查。"""
        signs = np.sign([s1, s2, s3])
        non_zero = [g for g in signs if g != 0]
        if len(non_zero) < min_agree:
            return 0
        cnt = Counter(non_zero)
        if cnt.most_common(1)[0][1] >= min_agree:
            return int(cnt.most_common(1)[0][0])
        return int(np.sign(np.sum(signs)))

    def crowding_protection(self, scores, current_score, base_th: float = 0.2):
        """根据同向排名抑制过度拥挤的信号，返回衰减系数。"""
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
            (ratio - self.core.max_same_direction_rate)
            / (1 - self.core.max_same_direction_rate),
        )
        rank_intensity = max(0.0, rank_pct - 0.8) / 0.2
        intensity = min(1.0, max(ratio_intensity, rank_intensity))

        factor = 1.0 - 0.2 * intensity
        dd = getattr(self.core, "_equity_drawdown", 0.0)
        factor *= max(0.6, 1 - dd)
        return factor

    def fuse(
        self,
        scores: dict,
        weights: tuple[float, float, float],
        strong_confirm_4h: bool,
    ) -> tuple[float, bool, bool, bool]:
        """按照多周期共振逻辑融合得分"""
        s1, s4, sd = scores['1h'], scores['4h'], scores['d1']
        w1, w4, wd = weights

        consensus_dir = self.consensus_check(s1, s4, sd)
        consensus_all = (
            consensus_dir != 0 and np.sign(s1) == np.sign(s4) == np.sign(sd)
        )
        consensus_14 = (
            consensus_dir != 0 and np.sign(s1) == np.sign(s4) and not consensus_all
        )
        consensus_4d1 = (
            consensus_dir != 0 and np.sign(s4) == np.sign(sd) and np.sign(s1) != np.sign(s4)
        )

        if consensus_all:
            fused = w1 * s1 + w4 * s4 + wd * sd
            conf = 1.1
            if strong_confirm_4h:
                conf *= 1.05
            fused *= self.core.cycle_weight.get("strong", 1.0)
        elif consensus_14:
            total = w1 + w4
            fused = (w1 / total) * s1 + (w4 / total) * s4
            conf = 0.9
            fused *= self.core.cycle_weight.get("weak", 1.0)
        elif consensus_4d1:
            total = w4 + wd
            fused = (w4 / total) * s4 + (wd / total) * sd
            conf = 0.9
            fused *= self.core.cycle_weight.get("weak", 1.0)
        else:
            fused = s1
            conf = 1.0

        fused_score = fused * conf
        penalty = 1.0
        if (
            np.sign(s1) != 0
            and (
                (np.sign(s4) != 0 and np.sign(s1) != np.sign(s4))
                or (np.sign(sd) != 0 and np.sign(s1) != np.sign(sd))
            )
        ):
            opp = self.core.cycle_weight.get("opposite", 1.0)
            fused_score *= opp
            penalty *= opp
        if not (consensus_all or consensus_14 or consensus_4d1):
            cm = getattr(self.core, "conflict_mult", 0.7)
            fused_score *= cm
            penalty *= cm
        logger.debug(
            "fuse scores s1=%.3f s4=%.3f sd=%.3f -> %.3f (conf=%.2f, penalty=%.2f)",
            s1,
            s4,
            sd,
            fused_score,
            conf,
            penalty,
        )
        return fused_score, consensus_all, consensus_14, consensus_4d1

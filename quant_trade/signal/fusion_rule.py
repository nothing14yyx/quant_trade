# -*- coding: utf-8 -*-
"""Rule-based fusion utilities extracted from RobustSignalGenerator."""

from __future__ import annotations

import numpy as np
from quant_trade.logging import get_logger
from .multi_period_fusion import (
    consensus_check as _consensus_check,
    crowding_protection as _crowding_protection,
    fuse_scores as _fuse_scores,
)

logger = get_logger(__name__)


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


class FusionRuleBased:
    """封装信号融合与拥挤度保护等规则逻辑。"""

    combine_score = staticmethod(combine_score)
    combine_score_vectorized = staticmethod(combine_score_vectorized)

    def __init__(self, core) -> None:
        self.core = core

    def consensus_check(self, s1, s2, s3, min_agree: int = 2):
        return _consensus_check(s1, s2, s3, min_agree)

    def crowding_protection(self, scores, current_score, base_th: float = 0.2):
        return _crowding_protection(
            scores,
            current_score,
            base_th,
            max_same_direction_rate=self.core.max_same_direction_rate,
            equity_drawdown=getattr(self.core, "_equity_drawdown", 0.0),
        )

    def fuse(
        self,
        scores: dict,
        weights: tuple[float, float, float],
        strong_confirm_4h: bool,
    ) -> tuple[float, bool, bool, bool]:
        return _fuse_scores(
            scores,
            weights,
            strong_confirm_4h,
            cycle_weight=self.core.cycle_weight,
            conflict_mult=getattr(self.core, "conflict_mult", 0.7),
        )

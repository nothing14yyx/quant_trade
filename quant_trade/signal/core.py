"""High level signal generation orchestrator.

此模块提供轻量级的 ``generate_signal`` 函数, 按既定顺序
调用各功能模块完成信号计算。原有的复杂实现被拆分至
``features_to_scores``、``ai_inference`` 等子模块中。本文件
仅负责流程调度, 以保证接口稳定。
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from . import features_to_scores, ai_inference, multi_period_fusion
from . import dynamic_thresholds, risk_filters, position_sizing


class AIModelPredictor:  # pragma: no cover - compatibility stub
    """Placeholder predictor to be monkeypatched in tests."""

    def __init__(self, model_paths):
        self.model_paths = model_paths

    def get_ai_score(self, *args, **kwargs):  # pragma: no cover
        return 0.0


def _safe_factor_scores(period_features: Mapping[str, Mapping[str, Any]]) -> dict:
    """Helper to obtain factor scores for each period."""
    scores: dict[str, Any] = {}
    for period, feats in period_features.items():
        try:
            # new API: get_factor_scores(features, period)
            scores[period] = features_to_scores.get_factor_scores(feats, period)
        except TypeError:
            # backward compat: get_factor_scores(core, features, period)
            from ..robust_signal_generator import RobustSignalGenerator

            class _Cache(dict):
                def set(self, k, v):
                    self[k] = v

            stub = RobustSignalGenerator.__new__(RobustSignalGenerator)
            stub._factor_cache = _Cache()
            scores[period] = features_to_scores.get_factor_scores(stub, feats, period)
    return scores


def generate_signal(
    features_1h: Mapping[str, Any],
    features_4h: Mapping[str, Any],
    features_d1: Mapping[str, Any],
    features_15m: Mapping[str, Any] | None = None,
    all_scores_list: list[float] | None = None,
    raw_features_1h: Mapping[str, Any] | None = None,
    raw_features_4h: Mapping[str, Any] | None = None,
    raw_features_d1: Mapping[str, Any] | None = None,
    raw_features_15m: Mapping[str, Any] | None = None,
    *,
    predictor: Any | None = None,
    models: Mapping[str, Mapping[str, Any]] | None = None,
    calibrators: Mapping[str, Mapping[str, Any]] | None = None,
    global_metrics: Mapping[str, Any] | None = None,
    open_interest: Mapping[str, Any] | None = None,
    order_book_imbalance: Mapping[str, Any] | None = None,
    symbol: str | None = None,
    ai_score_cache: Any | None = None,
    w_ai: float = 1.0,
    w_factor: float = 1.0,
    ic_stats: Mapping[str, float] | None = None,
    ic_threshold: float = 0.0,
) -> dict[str, Any]:
    """Generate trading signal using modular pipeline.

    该函数的参数与返回值保持与旧实现一致, 但内部逻辑已拆分
    为若干可单独测试的模块。调用顺序如下::

        factor_scores = features_to_scores.get_factor_scores(...)
        ai_scores, vol, rise, drawdown = ai_inference.get_period_ai_scores(...)
        fused_score = multi_period_fusion.fuse_scores(...)
        base_th, rev_boost = dynamic_thresholds.calc_dynamic_threshold(...)
        score_mult, pos_mult, reasons = risk_filters.compute_risk_multipliers(...)
        position, tp, sl = position_sizing.calc_position_size(...)
    """

    period_features = {"1h": features_1h, "4h": features_4h, "d1": features_d1}
    if features_15m is not None:
        period_features["15m"] = features_15m
    periods = ["1h", "4h", "d1"]

    # 1. 因子分
    factor_scores = _safe_factor_scores(period_features)

    # 2. AI 分与回归预测
    models = models or {}
    calibrators = calibrators or {}
    ai_scores = ai_inference.get_period_ai_scores(
        predictor, period_features, models, calibrators, cache=ai_score_cache
    )
    vol_preds, rise_preds, drawdown_preds = ai_inference.get_reg_predictions(
        predictor, period_features, models
    )

    # 根据 IC 或回测收益动态调整权重
    def _adjust_weight(weight: float, ic_val: float | None) -> float:
        if ic_val is None:
            return weight
        if ic_threshold > 0 and ic_val < ic_threshold:
            return weight * (ic_val / ic_threshold)
        return weight

    if ic_stats:
        w_ai = _adjust_weight(w_ai, ic_stats.get("ai"))
        w_factor = _adjust_weight(w_factor, ic_stats.get("factor"))

    # 将因子分与 AI 分按权重求和得到每周期的综合得分
    combined = {}
    for p in periods:
        fs = factor_scores.get(p, {})
        f_val = float(np.mean(list(fs.values()))) if fs else 0.0
        combined[p] = w_ai * ai_scores.get(p, 0.0) + w_factor * f_val

    # 3. 多周期融合
    ic_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ic_periods = {p: ic_stats.get(p) for p in periods} if ic_stats else None
    fused_score, consensus_all, consensus_14, consensus_4d1 = multi_period_fusion.fuse_scores(
        combined, ic_weights, False, ic_stats=ic_periods, min_agree=2
    )

    # 4. 动态阈值计算
    dyn_input = dynamic_thresholds.DynamicThresholdInput(
        atr=0.0,
        adx=0.0,
        history_scores=all_scores_list or [],
    )
    base_th, rev_boost = dynamic_thresholds.calc_dynamic_threshold(dyn_input)

    # 5. 风险乘数
    try:
        score_mult, pos_mult, reasons, risk_info = (
            risk_filters.compute_risk_multipliers(
                fused_score,
                base_th,
                combined,
                global_metrics=global_metrics,
                open_interest=open_interest,
                all_scores_list=all_scores_list,
                symbol=symbol,
            )
        )
    except Exception:  # pragma: no cover - graceful fallback
        score_mult, pos_mult, reasons, risk_info = 1.0, 1.0, [], {}

    # 6. 仓位与 TP/SL
    final_score = fused_score * score_mult
    pos_size = position_sizing.calc_position_size(
        final_score,
        base_th,
        max_position=pos_mult,
        min_exposure=0.2 * pos_mult,
    )
    take_profit, stop_loss = None, None

    details = {
        "factor_scores": factor_scores,
        "ai_scores": ai_scores,
        "vol_preds": vol_preds,
        "rise_preds": rise_preds,
        "drawdown_preds": drawdown_preds,
        "consensus_all": consensus_all,
        "consensus_14": consensus_14,
        "consensus_4d1": consensus_4d1,
        "risk_reasons": reasons,
        "base_th": risk_info.get("base_th", base_th),
        "rev_boost": rev_boost,
        "crowding_factor": risk_info.get("crowding_factor"),
        "oi_threshold": risk_info.get("oi_threshold"),
        "risk_score": risk_info.get("risk_score"),
        "flip": bool(risk_info.get("rev_dir")),
        "cooldown": risk_info.get("cooldown"),
    }

    return {
        "signal": int(np.sign(final_score)),
        "score": float(final_score),
        "position_size": float(pos_size),
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "details": details,
    }


__all__ = ["generate_signal"]


"""High level signal generation orchestrator.

此模块提供轻量级的 ``generate_signal`` 函数, 按既定顺序
调用各功能模块完成信号计算。原有的复杂实现被拆分至
``features_to_scores``、``ai_inference`` 等子模块中。本文件
仅负责流程调度, 以保证接口稳定。
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple
from pathlib import Path
import time

import numpy as np
import yaml

from . import features_to_scores, ai_inference, multi_period_fusion
from . import dynamic_thresholds, risk_filters, position_sizing
from ..config_manager import ConfigManager

# ---------------------------------------------------------------------------
# 权重与 IC 缓存
# ---------------------------------------------------------------------------

_WEIGHT_CACHE_PATH = Path("/tmp/quant_trade_weight_cache.yaml")
_FLUSH_INTERVAL = 600  # seconds
_last_weight_flush = 0.0

_cached_w_ai: dict[str, float] = {"1h": 1.0, "4h": 1.0, "d1": 1.0}
_cached_w_factor: dict[str, float] = {"1h": 1.0, "4h": 1.0, "d1": 1.0}


def refresh_weights(path: str | Path | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """外部模块可调用以刷新内部权重缓存."""
    p = Path(path) if path else _WEIGHT_CACHE_PATH
    if not p.exists():
        return _cached_w_ai, _cached_w_factor
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return _cached_w_ai, _cached_w_factor
    _cached_w_ai.update({k: float(v) for k, v in (data.get("w_ai") or {}).items()})
    _cached_w_factor.update({k: float(v) for k, v in (data.get("w_factor") or {}).items()})
    features_to_scores.category_ic.update(
        {k: float(v) for k, v in (data.get("category_ic") or {}).items()}
    )
    return _cached_w_ai, _cached_w_factor


def _maybe_flush_weights(w_ai: Mapping[str, float], w_factor: Mapping[str, float]) -> None:
    """定期将最新权重与 IC 写入缓存文件."""
    global _last_weight_flush
    now = time.time()
    if now - _last_weight_flush < _FLUSH_INTERVAL:
        return
    data = {"w_ai": dict(w_ai), "w_factor": dict(w_factor), "category_ic": features_to_scores.category_ic}
    try:
        with open(_WEIGHT_CACHE_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        _last_weight_flush = now
    except Exception:
        pass


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


def _load_fusion_cfg() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "signal.yaml"
    try:
        return ConfigManager(cfg_path).get("fusion", {}) or {}
    except Exception:
        return {}


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
    w_ai: float | Mapping[str, float] = 1.0,
    w_factor: float | Mapping[str, float] = 1.0,
    ic_stats: Mapping[str, float] | None = None,
    ic_threshold: float = 0.0,
    factor_weights: Mapping[str, float] | None = None,
    combine_score: Any | None = None,
    atr: float | None = None,
    adx: float | None = None,
    atr_4h: float | None = None,
    adx_4h: float | None = None,
    atr_d1: float | None = None,
    adx_d1: float | None = None,
    funding: float | None = None,
    backtest_returns: Mapping[str, float] | None = None,
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

    def _adjust_return(weight: float, ret: float | None) -> float:
        if ret is None:
            return weight
        return weight * (1.0 + ret)

    if isinstance(w_ai, Mapping):
        w_ai_periods = {p: float(w_ai.get(p, 1.0)) for p in periods}
    else:
        w_ai_periods = {p: float(w_ai) for p in periods}
    if isinstance(w_factor, Mapping):
        w_factor_periods = {p: float(w_factor.get(p, 1.0)) for p in periods}
    else:
        w_factor_periods = {p: float(w_factor) for p in periods}

    for p in periods:
        w_ai_periods[p] *= _cached_w_ai.get(p, 1.0)
        w_factor_periods[p] *= _cached_w_factor.get(p, 1.0)

    cat_weights = dict(factor_weights or {})
    if ic_stats:
        # 记录各类别 IC
        features_to_scores.record_ic({k: ic_stats.get(k) for k in cat_weights})
        for p in periods:
            w_ai_periods[p] = _adjust_weight(
                w_ai_periods[p], ic_stats.get(f"ai_{p}") or ic_stats.get("ai")
            )
            w_factor_periods[p] = _adjust_weight(
                w_factor_periods[p], ic_stats.get(f"factor_{p}") or ic_stats.get("factor")
            )
        cat_weights = {
            k: _adjust_weight(v, ic_stats.get(k)) for k, v in cat_weights.items()
        }
        _cached_w_ai.update(w_ai_periods)
        _cached_w_factor.update(w_factor_periods)
        _maybe_flush_weights(_cached_w_ai, _cached_w_factor)
    elif backtest_returns:
        for p in periods:
            w_ai_periods[p] = _adjust_return(
                w_ai_periods[p],
                backtest_returns.get(f"ai_{p}") or backtest_returns.get("ai"),
            )
            w_factor_periods[p] = _adjust_return(
                w_factor_periods[p],
                backtest_returns.get(f"factor_{p}") or backtest_returns.get("factor"),
            )
        cat_weights = {
            k: _adjust_return(v, backtest_returns.get(k))
            for k, v in cat_weights.items()
        }
        features_to_scores.record_ic(None)
        _cached_w_ai.update(w_ai_periods)
        _cached_w_factor.update(w_factor_periods)
        _maybe_flush_weights(_cached_w_ai, _cached_w_factor)
    else:
        features_to_scores.record_ic(None)

    # 将因子分与 AI 分按权重求和得到每周期的综合得分
    if callable(combine_score):
        combined = {
            p: float(
                combine_score(
                    ai_scores.get(p, 0.0), factor_scores.get(p, {}), cat_weights
                )
            )
            for p in periods
        }
    else:
        combined = {}
        for p in periods:
            fs = factor_scores.get(p, {})
            if fs:
                total_w = 0.0
                weighted = 0.0
                for k, v in fs.items():
                    w = cat_weights.get(k, 1.0)
                    weighted += w * v
                    total_w += w
                f_val = weighted / total_w if total_w else 0.0
            else:
                f_val = 0.0
            combined[p] = w_ai_periods[p] * ai_scores.get(p, 0.0) + w_factor_periods[p] * f_val

    # 3. 多周期融合
    ic_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ic_periods = {p: ic_stats.get(p) for p in periods} if ic_stats else None
    fusion_cfg = _load_fusion_cfg()
    fused_score, consensus_all, consensus_14, consensus_4d1 = multi_period_fusion.fuse_scores(
        combined,
        ic_weights,
        False,
        cycle_weight=fusion_cfg.get("cycle_weight"),
        conflict_mult=float(fusion_cfg.get("conflict_mult", 0.7)),
        ic_stats=ic_periods,
        min_agree=int(fusion_cfg.get("min_agree", 2)),
    )
    
    # 4. 动态阈值计算
    def _get_metric(feats, raw, *keys) -> float:
        for k in keys:
            if raw is not None and k in raw:
                try:
                    return float(raw[k])
                except Exception:
                    continue
            if feats is not None and k in feats:
                try:
                    return float(feats[k])
                except Exception:
                    continue
        return 0.0

    atr_val = float(atr) if atr is not None else _get_metric(
        features_1h, raw_features_1h, "atr_pct_1h", "atr_pct"
    )
    adx_val = float(adx) if adx is not None else _get_metric(
        features_1h, raw_features_1h, "adx_1h", "adx"
    )
    funding_val = float(funding) if funding is not None else _get_metric(
        features_1h, raw_features_1h, "funding_rate_1h", "funding_rate"
    )
    atr4_val = float(atr_4h) if atr_4h is not None else _get_metric(
        features_4h, raw_features_4h, "atr_pct_4h", "atr_pct"
    )
    adx4_val = float(adx_4h) if adx_4h is not None else _get_metric(
        features_4h, raw_features_4h, "adx_4h", "adx"
    )
    atrd_val = float(atr_d1) if atr_d1 is not None else _get_metric(
        features_d1, raw_features_d1, "atr_pct_d1", "atr_pct"
    )
    adxd_val = float(adx_d1) if adx_d1 is not None else _get_metric(
        features_d1, raw_features_d1, "adx_d1", "adx"
    )

    dyn_input = dynamic_thresholds.DynamicThresholdInput(
        atr=atr_val,
        adx=adx_val,
        funding=funding_val,
        atr_4h=atr4_val if atr4_val else None,
        adx_4h=adx4_val if adx4_val else None,
        atr_d1=atrd_val if atrd_val else None,
        adx_d1=adxd_val if adxd_val else None,
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
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "signal.yaml"
    try:
        min_exp_base = float(ConfigManager(cfg_path).get("min_exposure_base", 0.0))
    except Exception:  # pragma: no cover - fallback to zero
        min_exp_base = 0.0
    min_exposure = min_exp_base * pos_mult
    if isinstance(vol_preds, dict):
        vol_ref = vol_preds.get("1h") or vol_preds.get("4h")
        if vol_ref is not None and np.isfinite(vol_ref):
            vol_ref = min(max(abs(float(vol_ref)), 0.0), 1.0)
            min_exposure *= 1 - vol_ref
    if min_exposure <= 0:
        min_exposure = None
    pos_size = position_sizing.calc_position_size(
        final_score,
        base_th,
        max_position=pos_mult,
        min_exposure=min_exposure,
    )
    take_profit, stop_loss = None, None

    details = {
        "factor_scores": factor_scores,
        "ai_scores": ai_scores,
        "vol_preds": vol_preds,
        "rise_preds": rise_preds,
        "drawdown_preds": drawdown_preds,
        "scores": combined,
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


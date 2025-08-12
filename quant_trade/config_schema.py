from __future__ import annotations

"""Pydantic configuration schema for signal generation.

This module defines :class:`SignalConfig` which mirrors the configuration
structure expected by :class:`~quant_trade.signal.core.RobustSignalGenerator`.
Default values correspond to the project’s ``config.yaml`` so that
``SignalConfig().model_dump()`` returns a dictionary equivalent to the current
default configuration.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VoteSystem(BaseModel):
    """Parameters controlling the voting system."""

    weight_ai: float = 3
    strong_min: int = 2
    conf_min: float = 0.25
    prob_th: float = 0.5
    prob_margin: float = 0.1
    strong_prob_th: float = 0.8
    ai_dir_eps: float = 0.10


class DeltaBoost(BaseModel):
    """Δ-boost related configuration."""

    core_keys: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "1h": [
                "rsi_1h",
                "macd_hist_1h",
                "ema_diff_1h",
                "atr_pct_1h",
                "vol_ma_ratio_long_1h",
                "funding_rate_1h",
            ],
            "4h": [
                "rsi_4h",
                "macd_hist_4h",
                "ema_diff_4h",
            ],
            "d1": [
                "rsi_d1",
                "macd_hist_d1",
                "ema_diff_d1",
            ],
        }
    )
    params: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "rsi": [5, 1.0, 0.028899473694350645],
            "macd_hist": [0.002, 100.0, 0.028942909266482065],
            "ema_diff": [0.001, 100.0, 0.046287294659927015],
            "atr_pct": [0.002, 100.0, 0.02793080911149214],
            "vol_ma_ratio": [0.2, 1.0, 0.02557424610265277],
            "funding_rate": [0.0005, 10000, 0.04561010131390353],
        }
    )


class FeatureEngineering(BaseModel):
    rise_transform: str = "none"
    boxcox_lambda_path: str = "scalers/rise_boxcox_lambda.json"


class SignalThreshold(BaseModel):
    base_th: float = 0.09125270660476172
    gamma: float = 0.9
    min_pos: float = 0.05
    quantile: float = 0.78
    window: int = 60
    dynamic_quantile: float = 0.8
    low_base: float = 0.06
    rev_boost: float = 0.15
    rev_th_mult: float = 0.60


class DynamicThreshold(BaseModel):
    atr_mult: float = 3.0
    atr_cap: float = 0.10
    funding_mult: float = 6.0
    funding_cap: float = 0.08
    adx_div: float = 100.0
    adx_cap: float = 0.04
    smooth_window: int = 20
    smooth_alpha: float = 0.2
    smooth_limit: float = 1.0


class VoteWeights(BaseModel):
    ai: float = 2
    short_mom: float = 1
    vol_breakout: float = 1
    trend: float = 1
    confirm_15m: float = 1


class SignalFilters(BaseModel):
    min_vote: int = 1
    confidence_vote: float = 0.12
    conf_min: float = 0.25


class PositionCoeff(BaseModel):
    range: float = 0.40
    trend: float = 0.60
    low_vol_ratio: float = 0.3


class VolumeGuard(BaseModel):
    weak_scale: float = 0.93
    over_scale: float = 0.9
    ratio_low: float = 0.45
    ratio_high: float = 2.0
    roc_low: float = -20
    roc_high: float = 100
    volume_quantile_low: float = 0.2
    volume_quantile_high: float = 0.8


class ObThreshold(BaseModel):
    min_ob_th: float = 0.10
    dynamic_factor: float = 0.08


class OIProtection(BaseModel):
    scale: float = 0.9
    crowding_threshold: float = 0.98


class CycleWeight(BaseModel):
    strong: float = 1.2
    weak: float = 0.8
    opposite: float = 0.5
    conflict: float = 0.7


class Regime(BaseModel):
    adx_trend: int = 25
    adx_range: int = 20


class RiskAdjust(BaseModel):
    factor: float = 0.15


class ProtectionLimits(BaseModel):
    risk_score: float = 1.0


class TpSlEntry(BaseModel):
    tp_mult: float
    sl_mult: float


class TpSl(BaseModel):
    trend: TpSlEntry = TpSlEntry(tp_mult=1.8, sl_mult=1.2)
    range: TpSlEntry = TpSlEntry(tp_mult=1.0, sl_mult=0.8)
    sl_min_pct: float = 0.005


class SignalConfig(BaseModel):
    """Top-level configuration schema for signal generation."""

    risk_filters_enabled: bool = True
    dynamic_threshold_enabled: bool = True
    direction_filters_enabled: bool = False
    filter_penalty_mode: bool = True
    penalty_factor: float = 0.5
    rsi_k: float = 1.5
    enable_ai: bool = True
    enable_factor_breakdown: bool = False

    vote_system: VoteSystem = VoteSystem()
    delta_boost: DeltaBoost = DeltaBoost()
    feature_engineering: FeatureEngineering = FeatureEngineering()

    signal_threshold: SignalThreshold = SignalThreshold()
    dynamic_threshold: DynamicThreshold = DynamicThreshold()

    vote_weights: VoteWeights = VoteWeights()
    regime_vote_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "trend": {"trend": 2},
            "range": {"vol_breakout": 2},
        }
    )
    signal_filters: SignalFilters = SignalFilters()
    position_coeff: PositionCoeff = PositionCoeff()

    sentiment_alpha: float = 0.5
    cap_positive_scale: float = 0.85
    tp_sl: TpSl = TpSl()

    volume_guard: VolumeGuard = VolumeGuard()
    ob_threshold: ObThreshold = ObThreshold()
    exit_lag_bars: int = 1
    oi_protection: OIProtection = OIProtection()
    veto_level: float = 0.9
    veto_conflict_count: int = 1
    flip_coeff: float = 0.6
    flip_confirm_bars: int = 3
    cycle_weight: CycleWeight = CycleWeight()
    regime: Regime = Regime()

    risk_adjust: RiskAdjust = RiskAdjust()
    risk_adjust_threshold: Optional[float] = None
    risk_th_quantile: float = 0.6

    protection_limits: ProtectionLimits = ProtectionLimits()
    crowding_limit: float = 1.15
    max_position: float = 0.3
    risk_scale: float = 0.8
    min_pos_vol_scale: float = 0.0
    min_trend_align: int = 2
    th_down_d1: float = 0.74


__all__ = ["SignalConfig"]

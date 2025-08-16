import pytest

from collections import deque
import types

from quant_trade.utils.lru import LRU
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.risk_manager import RiskManager
from quant_trade.signal import (
    ThresholdingDynamic,
    PredictorAdapter,
    FactorScorerImpl,
    FusionRuleBased,
    RiskFiltersImpl,
    PositionSizerImpl,
)


def make_dummy_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.risk_manager = RiskManager()
    rsg._factor_cache = LRU(300)
    rsg._ai_score_cache = LRU(300)
    rsg.factor_scorer = FactorScorerImpl(rsg)
    rsg.history_scores = deque(maxlen=500)
    rsg.oi_change_history = deque(maxlen=500)
    rsg.models = {
        "1h": {"up": None, "down": None},
        "4h": {"up": None, "down": None},
        "d1": {"up": None, "down": None},
    }

    rsg.symbol_categories = {}

    rsg.max_same_direction_rate = 0.9
    rsg.base_weights = {
        'ai': 0.2, 'trend': 0.2, 'momentum': 0.2,
        'volatility': 0.2, 'volume': 0.1,
        'sentiment': 0.05, 'funding': 0.05,
    }
    rsg.ic_scores = {k: 1 for k in rsg.base_weights}
    rsg.current_weights = rsg.base_weights.copy()
    rsg._prev_raw = {p: None for p in ("15m", "1h", "4h", "d1")}
    rsg.vote_params = {'weight_ai': 2.0, 'strong_min': 5, 'conf_min': 1.0}
    rsg.min_weight_ratio = 0.2
    rsg.sentiment_alpha = 0.5
    rsg.cap_positive_scale = 0.4
    rsg.volume_guard_params = {
        'weak': 0.7,
        'over': 0.9,
        'ratio_low': 0.8,
        'ratio_high': 2.0,
        'roc_low': -20,
        'roc_high': 100,
    }
    rsg.ob_th_params = {'min_ob_th': 0.10, 'dynamic_factor': 0.08}
    rsg.regime_adx_trend = 25
    rsg.regime_adx_range = 20
    rsg.risk_adjust_factor = 0.9
    rsg.risk_adjust_threshold = 0
    rsg.risk_score_limit = 2.0
    rsg.crowding_limit = 1.1
    rsg.max_position = 0.3
    rsg.risk_scale = 1.0
    rsg.min_pos_vol_scale = 0.0
    rsg.volume_quantile_low = 0.2
    rsg.volume_quantile_high = 0.8
    rsg.volume_ratio_history = deque([0.8, 1.0, 1.2], maxlen=500)
    rsg.risk_filters_enabled = True
    rsg.dynamic_threshold_enabled = True
    rsg.enable_factor_breakdown = True
    rsg.cycle_weight = {'strong': 1.0, 'weak': 1.0, 'opposite': 1.0}
    rsg.smooth_window = 20
    rsg.smooth_alpha = 0.2
    rsg.smooth_limit = 1.0
    rsg.market_phase = "range"
    rsg.phase_dyn_mult = {}
    rsg.signal_params.rev_boost = 0.15
    cfg_dict = {
        'signal_threshold': {
            'mode': 'sigmoid',
            'base_th': 0.12,
            'gamma': 0.05,
            'min_pos': 0.10,
            'quantile': 0.80,
        },
        'ob_threshold': {'min_ob_th': 0.10},
        'risk_budget_per_trade': 0.01,
        'max_pos_pct': 0.3,
        'model_paths': {},
    }
    rsg.cfg = types.SimpleNamespace(**cfg_dict)
    rsg.cfg.get = lambda k, d=None: getattr(rsg.cfg, k, d)
    rsg.signal_threshold_cfg = rsg.cfg.signal_threshold
    rsg.min_trend_align = 1
    rsg.flip_confirm_bars = 3
    rsg.predictor = PredictorAdapter(None)
    rsg.fusion_rule = FusionRuleBased(rsg)
    rsg.consensus_check = rsg.fusion_rule.consensus_check
    rsg.crowding_protection = rsg.fusion_rule.crowding_protection
    rsg.fuse = rsg.fusion_rule.fuse
    rsg.fuse_multi_cycle = rsg.fusion_rule.fuse
    rsg.combine_score = rsg.fusion_rule.combine_score
    rsg.combine_score_vectorized = rsg.fusion_rule.combine_score_vectorized
    rsg.thresholding = ThresholdingDynamic(rsg)
    rsg.risk_filters = RiskFiltersImpl(rsg)
    rsg.position_sizer = PositionSizerImpl(rsg)
    rsg.dynamic_weight_update = lambda *a, **k: rsg.base_weights
    return rsg

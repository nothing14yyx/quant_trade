import pytest

from collections import deque
from quant_trade.robust_signal_generator import RobustSignalGenerator


def make_dummy_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.history_scores = deque(maxlen=500)
    rsg.oi_change_history = deque(maxlen=500)

    rsg.symbol_categories = {}

    rsg.max_same_direction_rate = 0.9
    rsg.base_weights = {
        'ai': 0.2, 'trend': 0.2, 'momentum': 0.2,
        'volatility': 0.2, 'volume': 0.1,
        'sentiment': 0.05, 'funding': 0.05,
    }
    rsg.ic_scores = {k: 1 for k in rsg.base_weights}
    rsg.current_weights = rsg.base_weights.copy()
    rsg._prev_raw = {p: None for p in ("1h", "4h", "d1")}
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
    rsg.risk_score_cap = 5.0
    rsg.regime_adx_trend = 25
    rsg.regime_adx_range = 20
    rsg.risk_adjust_factor = 0.9
    rsg.risk_adjust_threshold = -2.0
    rsg.risk_score_limit = 2.0
    rsg.crowding_limit = 1.1
    rsg.max_position = 0.3
    rsg.cycle_weight = {'strong': 1.0, 'weak': 1.0, 'opposite': 1.0}
    rsg.cfg = {
        'signal_threshold': {
            'mode': 'sigmoid',
            'base_th': 0.12,
            'gamma': 0.05,
            'min_pos': 0.10,
            'quantile': 0.80,
        },
        'ob_threshold': {'min_ob_th': 0.10},
    }
    rsg.signal_threshold_cfg = rsg.cfg['signal_threshold']
    return rsg

import numpy as np
import pytest
from collections import deque

from quant_trade.robust_signal_generator import RobustSignalGenerator, smooth_score


def make_rsg():
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
    rsg.vote_params = {'weight_ai': 2.0, 'strong_min': 3, 'conf_min': 1.0}
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
    rsg.signal_threshold_cfg = {
        'mode': 'sigmoid',
        'base_th': 0.12,
        'gamma': 0.05,
        'min_pos': 0.10,
        'rev_boost': 0.25,
        'rev_th_mult': 0.70,
        'low_base': 0.10,
    }
    rsg.flip_coeff = 0.3
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0.3
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, weights=None: ai
    rsg.dynamic_threshold = lambda *a, **k: (0.2, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg._raw_history = {'1h': deque(maxlen=4), '4h': deque(maxlen=2), 'd1': deque(maxlen=2)}
    rsg._cooldown = 3
    rsg.min_trend_align = 1
    rsg.flip_confirm_bars = 3
    return rsg


def test_detect_reversal():
    gen = RobustSignalGenerator.__new__(RobustSignalGenerator)
    prices = np.array([100, 95, 92, 98])
    assert gen.detect_reversal(prices, atr=0.02, volume=2.0) == 1


def test_flip_on_reversal():
    gen = make_rsg()
    gen.signal_threshold_cfg["min_pos"] = 0.0
    gen.last_signal = -1
    gen.last_score = -0.4
    gen._raw_history['1h'].extend([
        {'close': 95},
        {'close': 92},
        {'close': 90},
    ])
    feats_1h = {
        'close': 94,
        'atr_pct_1h': 0.02,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 2.0,
        'vol_breakout_1h': 1,
        'bb_width_1h': 0.02,
    }
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}
    res = gen.generate_signal(
        feats_1h,
        feats_4h,
        feats_d1,
        raw_features_1h=feats_1h,
    )
    assert res['signal'] == 0
    assert gen._cooldown == 0


def test_smooth_score_basic():
    hist = deque([1, 2, 3, 4])
    assert smooth_score(hist, window=3) == pytest.approx(3.0)


def test_flip_requires_confirmation():
    rsg = make_rsg()
    rsg.flip_confirm_bars = 3
    rsg._cooldown = 0
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, u, d: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, w=None: -0.4
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}
    rsg.risk_manager.calc_risk = lambda *a, **k: 0

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0.05,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 1,
        'vol_breakout_1h': 1,
        'bb_width_1h': 0.02,
    }
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    rsg._last_signal = 1
    rsg._last_score = 0.4

    # first bar, not enough confirmation
    assert rsg.generate_signal(feats_1h, feats_4h, feats_d1, raw_features_1h=feats_1h) is None
    assert rsg._last_signal == 1
    # second bar, still waiting
    assert rsg.generate_signal(feats_1h, feats_4h, feats_d1, raw_features_1h=feats_1h) is None
    assert rsg._last_signal == 1
    # third bar, flip allowed
    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1, raw_features_1h=feats_1h)
    assert res['signal'] == -1

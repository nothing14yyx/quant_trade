import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from collections import deque
import threading

from robust_signal_generator import RobustSignalGenerator


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.history_scores = deque(maxlen=500)
    rsg.oi_change_history = deque(maxlen=500)
    rsg.ic_history = {k: deque(maxlen=500) for k in ['ai','trend','momentum','volatility','volume','sentiment','funding']}
    rsg.symbol_categories = {}
    rsg.max_same_direction_rate = 0.9
    rsg.base_weights = {
        'ai': 0.2,
        'trend': 0.2,
        'momentum': 0.2,
        'volatility': 0.2,
        'volume': 0.1,
        'sentiment': 0.05,
        'funding': 0.05,
    }
    rsg.ic_scores = {k: 1 for k in rsg.base_weights}
    rsg.current_weights = rsg.base_weights.copy()
    rsg.vote_params = {'weight_ai': 2.0, 'strong_min': 5, 'conf_min': 1.0}
    rsg.min_weight_ratio = 0.2
    rsg._equity_drawdown = 0.0
    rsg._last_signal = 0
    rsg._last_score = 0.0
    rsg._lock = threading.RLock()
    rsg._prev_raw = {p: None for p in ("1h", "4h", "d1")}
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
    return rsg


def test_dynamic_weight_non_negative_sum_one():
    rsg = make_rsg()
    for k in rsg.ic_history:
        rsg.ic_history[k].extend([0.5, 1.0, -0.2])
    weights = rsg.dynamic_weight_update()
    assert all(w >= 0 for w in weights.values())
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0


def test_compute_tp_sl_fallback():
    rsg = make_rsg()
    tp, sl = rsg.compute_tp_sl(100, 0, 1)
    assert tp is not None and sl is not None


def test_flip_threshold_allows_switch():
    rsg = make_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, u, d: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, w=None: -0.75
    rsg.dynamic_threshold = lambda *a, **k: 0.6
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None}, '4h': {'up': None, 'down': None}, 'd1': {'up': None, 'down': None}}

    feats_1h = {'close': 100, 'atr_pct_1h': 0.05, 'adx_1h': 0, 'funding_rate_1h': 0, 'vol_ma_ratio_1h': 1}
    feats_4h = {'atr_pct_4h': 0.05}
    feats_d1 = {}
    rsg._last_signal = 1
    rsg._last_score = 0.1

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1, symbol='BTC')
    assert res['signal'] == -1


def test_range_filter_keeps_strong_signal():
    rsg = make_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, u, d: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, w=None: 0.6
    rsg.dynamic_threshold = lambda *a, **k: 0.5
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None}, '4h': {'up': None, 'down': None}, 'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0.05,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 0.2,
        'vol_roc_1h': -30,
    }
    feats_4h = {'atr_pct_4h': 0.05}
    feats_d1 = {}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1, raw_features_1h=feats_1h, symbol='BTC')
    assert res['signal'] == 0


def test_ma_cross_logic_symmetric():
    rsg = make_rsg()
    feats = {
        'sma_5_1h': 11,
        'sma_20_1h': 10,
        'ma_ratio_5_20': 1.03,
        'sma_20_1h_prev': 9.9,
    }
    assert rsg.ma_cross_logic(feats, feats['sma_20_1h_prev']) == pytest.approx(1.1)
    feats = {
        'sma_5_1h': 9.7,
        'sma_20_1h': 10,
        'ma_ratio_5_20': 0.97,
        'sma_20_1h_prev': 10.1,
    }
    assert rsg.ma_cross_logic(feats, feats['sma_20_1h_prev']) == pytest.approx(1.1)
    feats = {
        'sma_5_1h': 10,
        'sma_20_1h': 10,
        'ma_ratio_5_20': 1.01,
        'sma_20_1h_prev': 10,
    }
    assert rsg.ma_cross_logic(feats, feats['sma_20_1h_prev']) == pytest.approx(1.0)

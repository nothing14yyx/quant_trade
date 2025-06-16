import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from collections import deque

from robust_signal_generator import (
    RobustSignalGenerator,
    cap_positive,
    fused_to_risk,
)


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.history_scores = deque(maxlen=500)
    rsg.oi_change_history = deque(maxlen=500)
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
    rsg.ob_th_params = {'min_ob_th': 0.15, 'dynamic_factor': 0.08}
    rsg.risk_score_cap = 5.0
    rsg.exit_lag_bars = 2
    return rsg


def test_vol_roc_guard():
    rsg = make_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0.5
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: ai
    rsg.dynamic_threshold = lambda *a, **k: 0.0
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    f1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 1.0,
        'vol_roc_1h': -30,
    }
    f4h = {'atr_pct_4h': 0, 'adx_4h': 0, 'vol_ma_ratio_4h': 1.0, 'vol_roc_4h': -15}
    fd1 = {}

    res = rsg.generate_signal(f1h, f4h, fd1)
    assert res['details']['score_1h'] <= 0.35
    assert res['score'] * res['details']['score_1h'] >= 0


def test_score_clip():
    rsg = make_rsg()
    factor_scores = {
        'trend': -10,
        'momentum': -10,
        'volatility': 0,
        'volume': 0,
        'sentiment': -10,
        'funding': 0,
    }
    fused = rsg.combine_score(0, factor_scores)
    assert fused >= -1


def test_cap_positive():
    assert cap_positive(+0.6, -0.6, 0.4) == pytest.approx(+0.24)


def test_risk_cap():
    assert fused_to_risk(10, 0.1, 0.1, cap=5) == 5.0

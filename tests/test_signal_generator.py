import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from collections import deque

from robust_signal_generator import RobustSignalGenerator


def make_dummy_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.history_scores = deque(maxlen=500)
    rsg.max_same_direction_rate = 0.6
    return rsg


def test_compute_tp_sl():
    rsg = make_dummy_rsg()
    tp, sl = rsg.compute_tp_sl(100, 10, 1)
    assert tp == pytest.approx(115)
    assert sl == pytest.approx(90)

    tp, sl = rsg.compute_tp_sl(100, 10, -1)
    assert tp == pytest.approx(85)
    assert sl == pytest.approx(110)


def test_dynamic_threshold_basic():
    rsg = make_dummy_rsg()
    th = rsg.dynamic_threshold(0, 0, 0)
    assert th == pytest.approx(0.12)


def test_dynamic_threshold_upper_bound():
    rsg = make_dummy_rsg()
    th = rsg.dynamic_threshold(0.1, 50, 0.02)
    assert th == pytest.approx(0.25)


def test_consensus_check():
    rsg = make_dummy_rsg()
    assert rsg.consensus_check(0.2, 0.3, 0.1) == 1
    assert rsg.consensus_check(-0.2, -0.3, 0) == -1
    assert rsg.consensus_check(0.2, -0.3, 0) == 0


def test_crowding_protection():
    rsg = make_dummy_rsg()
    assert rsg.crowding_protection([1, 1, 1, 0, -1]) == 0
    assert rsg.crowding_protection([1, 1, -1, -1]) == 0


def test_generate_signal_raw_atr():
    rsg = make_dummy_rsg()

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
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, m: 0.9
    rsg.get_factor_scores = lambda f, p: {
        'trend': 1,
        'momentum': 1,
        'volatility': 1,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.9
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    features_1h = {'close': 100, 'atr_pct_1h': 0, 'adx_1h': 0, 'funding_rate_1h': 0}
    features_4h = {'atr_pct_4h': 0.1}
    features_d1 = {}
    raw_4h = {'atr_pct_4h': 0.05}

    res = rsg.generate_signal(features_1h, features_4h, features_d1, raw_features_4h=raw_4h)
    assert res['take_profit'] == pytest.approx(107.5)
    assert res['stop_loss'] == pytest.approx(95)

    res2 = rsg.generate_signal(features_1h, features_4h, features_d1)
    assert res2['take_profit'] == pytest.approx(115)
    assert res2['stop_loss'] == pytest.approx(90)


def test_factor_scores_use_raw_features():
    """确保提供 raw_features_* 时多因子计算使用原始数据"""
    rsg = make_dummy_rsg()

    captured = {}

    def fake_get_factor_scores(features, period):
        captured[period] = features
        return {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0,
                'sentiment': 0, 'funding': 0}

    rsg.get_factor_scores = fake_get_factor_scores
    rsg.get_ai_score = lambda f, m: 0
    rsg.combine_score = lambda ai, fs, weights=None: 0
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    feats_1h = {'atr_pct_1h': 0.1}
    feats_4h = {'atr_pct_4h': 0.1}
    feats_d1 = {'atr_pct_d1': 0.1}

    raw_1h = {'atr_pct_1h': 0.2}
    raw_4h = {'atr_pct_4h': 0.2}
    raw_d1 = {'atr_pct_d1': 0.2}

    rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                        raw_features_1h=raw_1h,
                        raw_features_4h=raw_4h,
                        raw_features_d1=raw_d1)

    assert captured['1h'] == raw_1h
    assert captured['4h'] == raw_4h
    assert captured['d1'] == raw_d1

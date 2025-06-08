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
    rsg.base_weights = {
        'ai': 0.2, 'trend': 0.2, 'momentum': 0.2,
        'volatility': 0.2, 'volume': 0.1,
        'sentiment': 0.05, 'funding': 0.05,
    }
    rsg.ic_scores = {k: 1 for k in rsg.base_weights}
    rsg.current_weights = rsg.base_weights.copy()
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
    assert th == pytest.approx(0.10)


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


def test_update_ic_scores_window_group(monkeypatch):
    rsg = make_dummy_rsg()

    import pandas as pd
    df = pd.DataFrame({
        "open_time": [0, 1, 0, 1],
        "open": [1, 1, 1, 1],
        "close": [1, 1, 1, 1],
        "symbol": ["A", "A", "B", "B"],
    })

    called = []

    def fake_compute_ic_scores(df_arg, rsg_arg):
        called.append(df_arg.copy())
        return {k: len(df_arg) for k in rsg.base_weights}

    import types, sys
    monkeypatch.setitem(sys.modules, "param_search", types.SimpleNamespace(
        compute_ic_scores=fake_compute_ic_scores
    ))

    rsg.update_ic_scores(df, window=1, group_by="symbol")

    assert len(called) == 2
    assert all(len(c) == 1 for c in called)
    assert rsg.ic_scores["ai"] == 1


def test_dynamic_weight_update(monkeypatch):
    rsg = make_dummy_rsg()

    import pandas as pd
    df = pd.DataFrame({"open_time": [0], "open": [1], "close": [1]})

    def fake_compute_ic_scores(df_arg, rsg_arg):
        return {k: i + 1 for i, k in enumerate(rsg.base_weights)}

    import types, sys
    monkeypatch.setitem(sys.modules, "param_search", types.SimpleNamespace(
        compute_ic_scores=fake_compute_ic_scores
    ))

    rsg.update_ic_scores(df)
    weights = rsg.dynamic_weight_update()

    ic_arr = np.array([max(0, v) for v in rsg.ic_scores.values()])
    base_arr = np.array([rsg.base_weights[k] for k in rsg.ic_scores])
    expected = ic_arr * base_arr / (ic_arr * base_arr).sum()

    assert weights["ai"] == pytest.approx(expected[0])
    assert rsg.current_weights == weights


def test_fused_score_not_extreme():
    """正常输入下 combine_score 不应给出極值"""
    rsg = make_dummy_rsg()

    factor_scores = {
        'trend': 0.8,
        'momentum': 0.7,
        'volatility': 0.6,
        'volume': 0.5,
        'sentiment': 0.1,
        'funding': 0.0,
    }

    fused = rsg.combine_score(0.8, factor_scores)
    assert abs(fused) < 1

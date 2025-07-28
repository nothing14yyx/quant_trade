import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.robust_signal_generator import RobustSignalGenerator, PeriodFeatures


def test_calc_factor_scores():
    rsg = make_dummy_rsg()
    ai = {'1h': 0.5, '4h': 0.2, 'd1': -0.1}
    fs = {
        '1h': {'trend': 0.1, 'momentum': 0, 'volatility': 0, 'volume': 0, 'sentiment': 0, 'funding': 0},
        '4h': {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0, 'sentiment': 0, 'funding': 0},
        'd1': {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0, 'sentiment': 0, 'funding': 0},
    }
    scores = rsg.calc_factor_scores(ai, fs, rsg.base_weights)
    w1 = rsg.base_weights.copy()
    for k in ('trend', 'momentum', 'volume'):
        w1[k] *= 0.7
    expected = rsg.combine_score(ai['1h'], fs['1h'], w1)
    assert scores['1h'] == pytest.approx(expected)


def test_apply_local_adjustments():
    rsg = make_dummy_rsg()
    scores = {'1h': 0.2, '4h': 0.2, 'd1': 0.2}
    raw = {
        '1h': {
            'sma_5_1h': 1.05,
            'sma_20_1h': 1.0,
            'ma_ratio_5_20': 1.05,
            'sma_20_1h_prev': 0.98,
            'vol_ma_ratio_1h': 2.0,
            'boll_perc_1h': 0.99,
            'vol_roc_1h': 0.0,
        },
        '4h': {'vol_ma_ratio_4h': 1.0, 'vol_roc_4h': 0.0},
        'd1': {'vol_ma_ratio_d1': 1.0, 'vol_roc_d1': 0.0},
    }
    fs = {
        '1h': {'sentiment': 0},
        '4h': {'trend': 1, 'momentum': 1, 'volatility': 1, 'sentiment': 0},
        'd1': {'sentiment': 0},
    }
    deltas = {'1h': {'rsi_1h_delta': 0.01}}
    adjusted, det = rsg.apply_local_adjustments(scores, raw, fs, deltas, 0.1, -0.05)
    assert 'ma_cross' in det
    assert 'strong_confirm_4h' in det
    assert 'boll_breakout_1h' in det
    assert adjusted['1h'] != scores['1h']


def test_rise_drawdown_adj_threshold():
    rsg = make_dummy_rsg()
    scores = {'1h': 0.2, '4h': 0.2, 'd1': 0.2}
    raw = {
        '1h': {
            'sma_5_1h': 1.05,
            'sma_20_1h': 1.0,
            'ma_ratio_5_20': 1.05,
            'sma_20_1h_prev': 0.98,
            'vol_ma_ratio_1h': 2.0,
            'boll_perc_1h': 0.99,
            'vol_roc_1h': 0.0,
        },
        '4h': {'vol_ma_ratio_4h': 1.0, 'vol_roc_4h': 0.0},
        'd1': {'vol_ma_ratio_d1': 1.0, 'vol_roc_d1': 0.0},
    }
    fs = {
        '1h': {'sentiment': 0},
        '4h': {'trend': 1, 'momentum': 1, 'volatility': 1, 'sentiment': 0},
        'd1': {'sentiment': 0},
    }
    deltas = {}
    adj_high, det_high = rsg.apply_local_adjustments(scores, raw, fs, deltas, 0.02, -0.005)
    assert det_high['rise_drawdown_adj'] > 0
    adj_low, det_low = rsg.apply_local_adjustments(scores, raw, fs, deltas, 0.008, -0.0)
    assert det_low['rise_drawdown_adj'] == 0


def test_fuse_multi_cycle():
    rsg = make_dummy_rsg()
    rsg.cycle_weight = {'strong': 2.0, 'weak': 0.5, 'opposite': 0.5}
    scores = {'1h': 0.2, '4h': 0.2, 'd1': 0.2}
    fused, a, b, c = rsg.fuse_multi_cycle(scores, (0.5, 0.3, 0.2), False)
    assert fused == pytest.approx(0.4)
    assert a and not b and not c
    scores = {'1h': 0.2, '4h': 0.2, 'd1': -0.1}
    fused2, a2, b2, c2 = rsg.fuse_multi_cycle(scores, (0.5, 0.3, 0.2), False)
    assert fused2 == pytest.approx(0.04)
    assert b2 and not a2
    scores = {'1h': -0.2, '4h': 0.2, 'd1': 0.2}
    fused3, a3, b3, c3 = rsg.fuse_multi_cycle(scores, (0.5, 0.3, 0.2), False)
    assert fused3 == pytest.approx(0.035)
    assert c3


def test_ai_dir_inconsistent_returns_none():
    rsg = make_dummy_rsg()
    rsg.ai_dir_eps = 0.1
    rsg.calc_factor_scores = lambda ai, fs, w: ai
    rsg.apply_local_adjustments = lambda s, *a, **k: (s, {})
    pf = PeriodFeatures({}, {})
    res = rsg.compute_factor_scores(
        {"1h": 0.3, "4h": -0.3, "d1": 0.3},
        pf,
        pf,
        pf,
        pf,
        {},
        {},
        {},
        {},
        None,
        None,
        None,
        None,
    )
    assert res is None


def test_ai_dir_eps_threshold_check():
    rsg = make_dummy_rsg()
    rsg.ai_dir_eps = 0.2
    rsg.calc_factor_scores = lambda ai, fs, w: ai
    rsg.apply_local_adjustments = lambda s, *a, **k: (s, {})
    pf = PeriodFeatures({}, {})
    res = rsg.compute_factor_scores(
        {"1h": 0.15, "4h": -0.3, "d1": 0.25},
        pf,
        pf,
        pf,
        pf,
        {},
        {},
        {},
        {},
        None,
        None,
        None,
        None,
    )
    assert res is None


def test_compute_exit_multiplier():
    rsg = make_dummy_rsg()
    rsg._exit_lag = 0
    assert rsg.compute_exit_multiplier(2, 5, 1) == 0.5
    assert rsg._exit_lag == 0
    val = rsg.compute_exit_multiplier(-1, 2, 1)
    assert val == 0.0
    assert rsg._exit_lag == 1

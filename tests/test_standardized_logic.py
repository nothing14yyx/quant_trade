import pytest
from quant_trade.tests.test_utils import make_dummy_rsg


def test_range_filter_respects_vol_breakout_zero():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.predictor.get_ai_score = lambda f, up, down: 0
    rsg.factor_scorer.score = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, weights=None: 0.6
    rsg.dynamic_threshold = lambda *a, **k: (0.5, 0.0)
    rsg.position_sizer.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    f1h = {
        'close': 100,
        'atr_pct_1h': 0.01,
        'adx_1h': 10,
        'funding_rate_1h': 0,
        'vol_breakout_1h': 0,
        'bb_width_1h': 0.02,
    }
    f4h = {'atr_pct_4h': 0.01, 'adx_4h': 10}
    fd1 = {}

    res = rsg.generate_signal(
        f1h,
        f4h,
        fd1,
        raw_features_1h=f1h,
        raw_features_4h=f4h,
        raw_features_d1=fd1,
    )

    assert res['signal'] == 0


def test_break_support_triggers_range_when_standardized():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.predictor.get_ai_score = lambda f, up, down: 0
    rsg.factor_scorer.score = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, weights=None: 0.6
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.position_sizer.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    f1h = {
        'close': 100,
        'atr_pct_1h': 0.01,
        'adx_1h': 30,
        'funding_rate_1h': 0,
        'vol_breakout_1h': 1,
        'bb_width_1h': 0.02,
    }
    f4h = {'atr_pct_4h': 0.01, 'adx_4h': 30}
    fd1 = {
        'atr_pct_d1': 0.01,
        'adx_d1': 30,
        'rsi_d1': 25,
        'break_support_d1': 0.5,
    }

    res = rsg.generate_signal(
        f1h,
        f4h,
        fd1,
        raw_features_1h=f1h,
        raw_features_4h=f4h,
        raw_features_d1=fd1,
    )

    assert res['details']['exit']['regime'] == 'range'

import pytest
from collections import deque
from robust_signal_generator import RobustSignalGenerator


def make_rsg():
    r = RobustSignalGenerator.__new__(RobustSignalGenerator)
    r.history_scores = deque(maxlen=500)
    r.oi_change_history = deque(maxlen=500)
    r.symbol_categories = {}
    r.max_same_direction_rate = 0.9
    r.base_weights = {
        'ai': 0.2,
        'trend': 0.2,
        'momentum': 0.2,
        'volatility': 0.2,
        'volume': 0.1,
        'sentiment': 0.05,
        'funding': 0.05,
    }
    r.ic_scores = {k: 1 for k in r.base_weights}
    r.current_weights = r.base_weights.copy()
    r._prev_raw = {p: None for p in ("1h", "4h", "d1")}
    return r


def test_range_guard():
    gen = make_rsg()
    gen.dynamic_weight_update = lambda: gen.base_weights
    gen.get_ai_score = lambda f, up, down: 0.0
    scores_seq = iter([0.55, -0.75, 0.0])
    gen.combine_score = lambda ai, fs, weights=None: next(scores_seq)
    gen.get_factor_scores = lambda f, p: {k: 0 for k in gen.base_weights if k != 'ai'}
    gen.dynamic_threshold = lambda *a, **k: 0.1
    gen.compute_tp_sl = lambda *a, **k: (0, 0)
    gen.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    row_1h = {
        'close': 100,
        'atr_pct_1h': 0.01,
        'adx_1h': 10,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 1.0,
        'vol_roc_1h': 150,
    }
    row_4h = {
        'atr_pct_4h': 0.01,
        'adx_4h': 10,
        'vol_ma_ratio_4h': 1.0,
        'vol_roc_4h': -5,
    }
    row_d1 = {'atr_pct_d1': 0.01, 'adx_d1': 10}

    res = gen.generate_signal(row_1h, row_4h, row_d1,
                              raw_features_1h=row_1h,
                              raw_features_4h=row_4h,
                              raw_features_d1=row_d1)
    sig = res['signal']
    assert sig in (0, -1)
    if sig == 1:
        pytest.fail("Range guard 未生效，应禁止做多")


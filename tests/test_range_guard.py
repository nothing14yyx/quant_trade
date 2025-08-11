import pytest
from collections import deque
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal.predictor_adapter import PredictorAdapter


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
    r.sentiment_alpha = 0.5
    r.cap_positive_scale = 0.4
    r.volume_guard_params = {
        'weak': 0.7,
        'over': 0.9,
        'ratio_low': 0.8,
        'ratio_high': 2.0,
        'roc_low': -20,
        'roc_high': 100,
    }
    r.ob_th_params = {'min_ob_th': 0.10, 'dynamic_factor': 0.08}
    r.regime_adx_trend = 25
    r.regime_adx_range = 20
    r.risk_adjust_factor = 0.9
    r.risk_adjust_threshold = 0
    r.risk_score_limit = 2.0
    r.crowding_limit = 1.1
    r.max_position = 0.3
    r.risk_scale = 1.0
    r.min_pos_vol_scale = 0.0
    r.volume_quantile_low = 0.2
    r.volume_quantile_high = 0.8
    r.volume_ratio_history = deque([0.8, 1.0, 1.2], maxlen=500)
    r.flip_confirm_bars = 3
    r.predictor = PredictorAdapter(None)
    return r


def test_range_guard():
    gen = make_rsg()
    gen.dynamic_weight_update = lambda: gen.base_weights
    gen.predictor.get_ai_score = lambda f, up, down: 0.0
    scores_seq = iter([0.55, -0.75, 0.0])
    gen.combine_score = lambda ai, fs, weights=None: next(scores_seq)
    gen.get_factor_scores = lambda f, p: {k: 0 for k in gen.base_weights if k != 'ai'}
    gen.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
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
        'vol_roc_1h': int(gen.volume_guard_params['roc_high'] * 1.5),
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


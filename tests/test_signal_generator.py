import numpy as np
import pytest
from collections import deque

from quant_trade.robust_signal_generator import RobustSignalGenerator, sigmoid_dir

def compute_vix_proxy(fr, oi):
    return 0.5 * fr + 0.5 * oi


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
    rsg.cycle_weight = {'strong': 1.0, 'weak': 1.0, 'opposite': 1.0}
    rsg.cfg = {
        'signal_threshold': {
            'mode': 'sigmoid',
            'base_th': 0.12,
            'gamma': 0.05,
            'min_pos': 0.10,
        },
        'ob_threshold': {'min_ob_th': 0.10},
    }
    rsg.signal_threshold_cfg = rsg.cfg['signal_threshold']
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
    th, _ = rsg.dynamic_threshold(0, 0, 0)
    assert th == pytest.approx(0.12)


def test_get_dynamic_oi_threshold():
    rsg = make_dummy_rsg()
    rsg.oi_change_history.extend([0.1]*80 + [0.6]*20)
    th = rsg.get_dynamic_oi_threshold()
    assert th == pytest.approx(0.6)
    th2 = rsg.get_dynamic_oi_threshold(pred_vol=0.2)
    assert th2 > th


def test_dynamic_threshold_upper_bound():
    rsg = make_dummy_rsg()
    th, _ = rsg.dynamic_threshold(0.1, 50, 0.02)
    assert th == pytest.approx(0.47)


def test_dynamic_threshold_multi_period():
    rsg = make_dummy_rsg()
    th1, _ = rsg.dynamic_threshold(0.02, 25)
    th2, _ = rsg.dynamic_threshold(0.02, 25, atr_4h=0.01, adx_4h=25)
    th3, _ = rsg.dynamic_threshold(0.02, 25, atr_4h=0.01, adx_4h=25, atr_d1=0.01, adx_d1=25)

    assert th1 == pytest.approx(0.24)
    assert th2 == pytest.approx(0.2625)
    assert th3 == pytest.approx(0.27375)


def test_dynamic_threshold_with_vix():
    rsg = make_dummy_rsg()
    base_th, _ = rsg.dynamic_threshold(0.02, 25)
    th, _ = rsg.dynamic_threshold(0.02, 25, vix_proxy=1.0)
    assert th > base_th


def test_dynamic_threshold_with_computed_vix():
    rsg = make_dummy_rsg()
    proxy = compute_vix_proxy(0.02, 0.04)
    base_th, _ = rsg.dynamic_threshold(0.02, 25)
    th, _ = rsg.dynamic_threshold(0.02, 25, vix_proxy=proxy)
    assert th > base_th


def test_dynamic_threshold_recovery():
    rsg = make_dummy_rsg()
    rsg.th_window = 50
    rsg.history_scores.extend([5.0] * 120)
    th_high, _ = rsg.dynamic_threshold(0, 0, 0)
    assert th_high >= 5
    for _ in range(60):
        rsg.history_scores.append(0.1)
    th_normal, _ = rsg.dynamic_threshold(0, 0, 0)
    assert th_normal == pytest.approx(0.12)


def test_consensus_check():
    rsg = make_dummy_rsg()
    assert rsg.consensus_check(0.2, 0.3, 0.1) == 1
    assert rsg.consensus_check(-0.2, -0.3, 0) == -1
    assert rsg.consensus_check(0.2, -0.3, 0) == 0


def test_crowding_protection():
    rsg = make_dummy_rsg()
    factor = rsg.crowding_protection([0.9, 0.8, 0.85, -0.2]*18, 0.95, base_th=0.2)
    assert factor == pytest.approx(0.5)
    factor2 = rsg.crowding_protection([0.1, -0.2]*15, 0.15, base_th=0.2)
    assert factor2 == pytest.approx(1.0)


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
    rsg.get_ai_score = lambda f, up, down: 0.9
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

    res = rsg.generate_signal(features_1h, features_4h, features_d1, raw_features_4h=raw_4h, symbol="BTCUSDT")
    assert res['take_profit'] is None
    assert res['stop_loss'] is None

    res2 = rsg.generate_signal(features_1h, features_4h, features_d1, symbol="BTCUSDT")
    assert res2['take_profit'] is None
    assert res2['stop_loss'] is None


def test_factor_scores_use_normalized_features():
    """确保多因子评分始终使用标准化后的特征"""
    rsg = make_dummy_rsg()

    captured = {}

    def fake_get_factor_scores(features, period):
        captured[period] = features
        return {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0,
                'sentiment': 0, 'funding': 0}

    rsg.get_factor_scores = fake_get_factor_scores
    rsg.get_ai_score = lambda f, up, down: 0
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

    rsg.generate_signal(
        feats_1h,
        feats_4h,
        feats_d1,
        raw_features_1h=raw_1h,
        raw_features_4h=raw_4h,
        raw_features_d1=raw_d1,
    )

    assert captured['1h'] == feats_1h
    assert captured['4h'] == feats_4h
    assert captured['d1'] == feats_d1


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
    monkeypatch.setitem(sys.modules, "quant_trade.param_search", types.SimpleNamespace(
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
    monkeypatch.setitem(sys.modules, "quant_trade.param_search", types.SimpleNamespace(
        compute_ic_scores=fake_compute_ic_scores
    ))

    rsg.update_ic_scores(df)
    weights = rsg.dynamic_weight_update()

    base_arr = np.array(list(rsg.base_weights.values()))
    ic_arr = np.array(list(rsg.ic_scores.values()))
    expected_raw = base_arr * (1 + ic_arr)
    expected = expected_raw / expected_raw.sum()

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


def test_combine_score_weight_names():
    """确保 combine_score 使用因子名称对应权重"""
    rsg = make_dummy_rsg()

    ai = 0.1
    factor_scores = {
        'trend': 0.2,
        'momentum': 0.3,
        'volatility': 0.4,
        'volume': 0.5,
        'sentiment': 0.6,
        'funding': 0.7,
    }

    weights = {
        'funding': 0.05,
        'sentiment': 0.05,
        'volume': 0.1,
        'volatility': 0.2,
        'momentum': 0.2,
        'trend': 0.2,
        'ai': 0.2,
    }

    expected = (
        ai * weights['ai']
        + factor_scores['trend'] * weights['trend']
        + factor_scores['momentum'] * weights['momentum']
        + factor_scores['volatility'] * weights['volatility']
        + factor_scores['volume'] * weights['volume']
        + factor_scores['sentiment'] * weights['sentiment']
        + factor_scores['funding'] * weights['funding']
    )

    fused = rsg.combine_score(ai, factor_scores, weights)
    assert fused == pytest.approx(expected)


def test_generate_signal_with_external_metrics():
    base = make_dummy_rsg()
    for r in (base,):
        r.get_ai_score = lambda f, up, down: 0
        r.get_factor_scores = lambda f, p: {
            'trend': 0,
            'momentum': 0,
            'volatility': 0,
            'volume': 0,
            'sentiment': 0,
            'funding': 0,
        }
        r.combine_score = lambda ai, fs, weights=None: 0.5
        r.dynamic_weight_update = lambda: r.base_weights
        r.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
        r.compute_tp_sl = lambda *a, **k: (0, 0)
        r.models = {'1h': {'up': None, 'down': None},
                    '4h': {'up': None, 'down': None},
                    'd1': {'up': None, 'down': None}}

    feats_1h = {'close': 100, 'atr_pct_1h': 0, 'adx_1h': 0, 'funding_rate_1h': 0}
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    baseline = base.generate_signal(feats_1h, feats_4h, feats_d1, symbol="BTCUSDT")
    assert baseline['score'] == pytest.approx(0.5)

    rsg = make_dummy_rsg()
    rsg.get_ai_score = base.get_ai_score
    rsg.get_factor_scores = base.get_factor_scores
    rsg.combine_score = base.combine_score
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = base.dynamic_threshold
    rsg.compute_tp_sl = base.compute_tp_sl
    rsg.models = base.models

    gm = {'btc_dom_chg': 0.1, 'mcap_growth': 0.1, 'vol_chg': 0.1}
    oi = {'oi_chg': 0.1}
    result = rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                                 global_metrics=gm, open_interest=oi, symbol="BTCUSDT")
    expected = result['score']
    assert result['score'] == pytest.approx(expected)


def test_hot_sector_influence():
    rsg = make_dummy_rsg()
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.5
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    rsg.symbol_categories = {'ABC': ['Gaming', 'DeFi']}

    feats_1h = {'close': 100, 'atr_pct_1h': 0, 'adx_1h': 0, 'funding_rate_1h': 0}
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    gm = {'hot_sector_strength': 0.2, 'hot_sector': 'Gaming'}

    result = rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                                 global_metrics=gm, symbol='ABC')
    expected = 0.5 * (1 + 0.05 * 0.2)
    assert result['score'] == pytest.approx(expected)


def test_eth_dominance_influence():
    rsg = make_dummy_rsg()
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.5
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    # Seed dominance history to enable diff calculation
    rsg.eth_dom_history.extend([0, 0, 0, 0, 0])

    feats_1h = {'close': 100, 'atr_pct_1h': 0, 'adx_1h': 0, 'funding_rate_1h': 0}
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    gm = {'eth_dominance': 0.1}

    result = rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                                 global_metrics=gm, symbol='ETHUSDT')
    expected = 0.5 * (1 + 0.1 * 0.2)
    assert result['score'] == pytest.approx(expected)


def test_short_momentum_and_order_book():
    rsg = make_dummy_rsg()
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.5
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'mom_5m_roll1h': 0.1,
        'mom_15m_roll1h': 0.1,
        'bid_ask_imbalance': 0.1,
    }
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1)
    assert res['score'] > 0.5
    assert res['details']['short_momentum'] > 0
    assert res['details']['ob_imbalance'] > 0


def test_ma_cross_logic_amplify():
    rsg = make_dummy_rsg()
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.5
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'sma_5_1h': 10.3,
        'sma_20_1h': 10,
        'ma_ratio_5_20': 1.03,
        'sma_20_1h_prev': 9.8,
    }
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1, raw_features_1h=feats_1h)
    assert res['score'] > 0.5
    assert res['details']['ma_cross'] == 1


def test_ma_cross_logic_overbought_threshold():
    rsg = make_dummy_rsg()
    feats = {
        'sma_5_1h': 11.2,
        'sma_20_1h': 10,
        'ma_ratio_5_20': 1.12,
        'sma_20_1h_prev': 9.9,
    }
    assert rsg.ma_cross_logic(feats, feats['sma_20_1h_prev']) == pytest.approx(1.15)
    feats['ma_ratio_5_20'] = 1.06
    assert rsg.ma_cross_logic(feats, feats['sma_20_1h_prev']) == pytest.approx(1.15)


def test_dynamic_threshold_regime():
    rsg = make_dummy_rsg()
    base = rsg.dynamic_threshold(0.02, 25)
    th_trend = rsg.dynamic_threshold(0.02, 25, regime='trend')
    th_range = rsg.dynamic_threshold(0.02, 25, regime='range')
    assert th_trend > base
    assert th_range < base
    assert th_trend > th_range


def test_order_book_momentum_threshold():
    """小幅盘口差异不应取消已生成的信号"""
    rsg = make_dummy_rsg()
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: 0.5
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {
        '1h': {'up': None, 'down': None},
        '4h': {'up': None, 'down': None},
        'd1': {'up': None, 'down': None},
    }

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0.05,
        'adx_1h': 0,
        'funding_rate_1h': 0,
    }
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    res = rsg.generate_signal(
        feats_1h, feats_4h, feats_d1, order_book_imbalance=-0.01
    )
    assert res['signal'] == 0


def test_sentiment_reweight_and_guard():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0

    def fs_func(feats, period):
        if period == '1h':
            return {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0,
                    'sentiment': -0.6, 'funding': 0}
        if period == '4h':
            return {'trend': 1, 'momentum': 0, 'volatility': 0, 'volume': 0,
                    'sentiment': 0, 'funding': 0}
        return {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0,
                'sentiment': 0, 'funding': 0}

    rsg.get_factor_scores = fs_func
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {'close': 100, 'atr_pct_1h': 0, 'adx_1h': 0,
                'funding_rate_1h': 0, 'vol_ma_ratio_1h': 1.0}
    feats_4h = {'atr_pct_4h': 0, 'adx_4h': 0}
    feats_d1 = {'atr_pct_d1': 0, 'adx_d1': 0}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1)
    scores = res['details']['scores']
    assert scores['1h'] == pytest.approx(-0.03899, rel=1e-3)
    assert scores['4h'] == pytest.approx(0.2, rel=1e-3)


def test_volume_and_funding_penalties():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0

    def fs_func(feats, period):
        return {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0,
                'sentiment': 0, 'funding': 0}

    rsg.get_factor_scores = fs_func
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 0.7,
        'vol_roc_1h': -0.25,
        'supertrend_dir_1h': 1,
        'funding_rate_anom_1h': -0.02,
    }
    feats_4h = {'atr_pct_4h': 0, 'adx_4h': 0}
    feats_d1 = {'atr_pct_d1': 0, 'adx_d1': 0}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1)
    assert res['details']['scores']['1h'] == pytest.approx(0.0)


def test_momentum_alignment_disables_confirm():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0

    def fs_func(feats, period):
        if period == '4h':
            return {'trend': 1, 'momentum': 1, 'volatility': 1,
                    'volume': 0, 'sentiment': 0, 'funding': 0}
        return {'trend': 0, 'momentum': 0, 'volatility': 0,
                'volume': 0, 'sentiment': 0, 'funding': 0}

    rsg.get_factor_scores = fs_func
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'macd_hist_diff_1h_4h': -0.1,
        'rsi_diff_1h_4h': -9,
    }
    feats_4h = {'atr_pct_4h': 0, 'adx_4h': 0}
    feats_d1 = {'atr_pct_d1': 0, 'adx_d1': 0}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1)
    assert res['details'].get('strong_confirm_vote', False) is False
    assert res['details'].get('vote', {}).get('value', 0) < 5


def test_crowding_factor_and_dynamic_threshold():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0

    def fs_func(feats, period):
        return {'trend': 0, 'momentum': 0, 'volatility': 0,
                'volume': 0, 'sentiment': -0.6 if period == '1h' else 0,
                'funding': 0}

    rsg.get_factor_scores = fs_func
    rsg.get_dynamic_oi_threshold = lambda pred_vol=None: 0.6
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 0.7,
    }
    feats_4h = {'atr_pct_4h': 0, 'adx_4h': 0}
    feats_d1 = {'atr_pct_d1': 0, 'adx_d1': 0}

    oi = {'oi_chg': 0}
    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                              open_interest=oi)
    assert res['details']['exit']['dynamic_th_final'] == pytest.approx(0.1)
    env = res['details']['env']
    expected = env['logic_score'] * env['env_score'] * env['risk_score']
    assert res['score'] == pytest.approx(expected)


def test_step_exit_with_order_book_flip():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda f, up, down: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, weights=None: 0.6
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    f1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'mom_5m_roll1h': 0.1,
        'mom_15m_roll1h': 0.1,
        'vol_breakout_1h': 1,
        'vol_ratio_1h_4h': 1.0,
    }
    f4h = {'atr_pct_4h': 0, 'adx_4h': 0, 'vol_ratio_1h_4h': 1.0}
    fd1 = {}

    res1 = rsg.generate_signal(f1h, f4h, fd1, order_book_imbalance=0.3)
    assert res1['signal'] == 0

    res2 = rsg.generate_signal(f1h, f4h, fd1, order_book_imbalance=-0.3)
    assert res2['signal'] == 0
    assert res2['position_size'] == res1['position_size']


def test_position_size_range_regime():
    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights

    def fake_ai(feats, up, down):
        if 'atr_pct_1h' in feats:
            return 0.5
        return -0.2

    rsg.get_ai_score = fake_ai
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0, 'momentum': 0, 'volatility': 0,
        'volume': 0, 'sentiment': 0, 'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: ai
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    f1h = {
        'close': 100,
        'atr_pct_1h': 0.01,
        'adx_1h': 10,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 1.0,
    }
    f4h = {'atr_pct_4h': 0.01, 'adx_4h': 15}
    fd1 = {'atr_pct_d1': 0.01, 'adx_d1': 20}

    res = rsg.generate_signal(f1h, f4h, fd1,
                              raw_features_1h=f1h,
                              raw_features_4h=f4h,
                              raw_features_d1=fd1)
    assert res['position_size'] == 0


def test_generate_signal_with_cls_model():
    rsg = make_dummy_rsg()

    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score_cls = lambda feats, mdl: 0.5
    rsg.get_factor_scores = lambda f, p: {
        'trend': 0,
        'momentum': 0,
        'volatility': 0,
        'volume': 0,
        'sentiment': 0,
        'funding': 0,
    }
    rsg.combine_score = lambda ai, fs, weights=None: ai
    rsg.dynamic_threshold = lambda *a, **k: (0.0, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.ma_cross_logic = lambda *a, **k: 1.0
    rsg.models = {
        '1h': {'cls': None},
        '4h': {'cls': None},
        'd1': {'cls': None},
    }

    f1h = {
        'close': 100,
        'atr_pct_1h': 0,
        'adx_1h': 0,
        'funding_rate_1h': 0,
        'vol_ma_ratio_1h': 1.0,
    }
    f4h = {'atr_pct_4h': 0}
    fd1 = {}

    res = rsg.generate_signal(f1h, f4h, fd1)
    assert res['score'] == pytest.approx(0.5)


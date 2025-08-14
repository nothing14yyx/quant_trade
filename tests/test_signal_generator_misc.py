import pytest
import numpy as np
from collections import deque

from quant_trade.utils.lru import LRU

from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.market_phase import get_market_phase
from quant_trade.signal import FactorScorerImpl


def make_rsg():
    r = RobustSignalGenerator.__new__(RobustSignalGenerator)
    r._factor_cache = LRU(300)
    r._ai_score_cache = LRU(300)
    r.factor_scorer = FactorScorerImpl(r)
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
    return r


def test_detect_market_regime():
    res = get_market_phase(
        None,
        {
            "adx1": 30,
            "adx4": 20,
            "adxd": 25,
            "bb_width_chg": 0.1,
            "channel_pos": 0.6,
        },
    )
    assert res["regime"] == "trend"

    res = get_market_phase(
        None,
        {
            "adx1": 10,
            "adx4": 15,
            "adxd": 20,
            "bb_width_chg": -0.1,
            "channel_pos": 0.2,
        },
    )
    assert res["regime"] == "range"


def test_detect_market_regime_all_nan():
    res = get_market_phase(
        None,
        {
            "adx1": np.nan,
            "adx4": np.nan,
            "adxd": np.nan,
            "bb_width_chg": None,
            "channel_pos": None,
        },
    )
    assert res["regime"] == "range"


def test_get_ic_period_weights():
    rsg = make_rsg()
    ic = {"1h": 1.0, "4h": 0.5, "d1": 0.2}
    w1, w4, wd = rsg.get_ic_period_weights(ic)
    base = np.array([3, 2, 1], dtype=float)
    ic_arr = np.array([1.0, 0.5, 0.2])
    expected = base * ic_arr
    expected /= expected.sum()
    assert w1 == pytest.approx(expected[0])
    assert w4 == pytest.approx(expected[1])
    assert wd == pytest.approx(expected[2])
    assert pytest.approx(w1 + w4 + wd) == 1.0

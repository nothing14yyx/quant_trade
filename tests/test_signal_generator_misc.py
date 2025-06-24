import pytest
import numpy as np

from quant_trade.robust_signal_generator import RobustSignalGenerator


def make_rsg():
    r = RobustSignalGenerator.__new__(RobustSignalGenerator)
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
    r.risk_score_cap = 5.0
    return r


def test_detect_market_regime():
    rsg = make_rsg()
    assert rsg.detect_market_regime(30, 20, 25) == "trend"
    assert rsg.detect_market_regime(10, 15, 20) == "range"


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

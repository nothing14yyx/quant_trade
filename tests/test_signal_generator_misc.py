import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from robust_signal_generator import RobustSignalGenerator


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
    r.ob_th_params = {'min_ob_th': 0.15, 'dynamic_factor': 0.08}
    r.risk_score_cap = 5.0
    r.exit_lag_bars = 2
    return r


def test_detect_market_regime():
    rsg = make_rsg()
    assert rsg.detect_market_regime(30, 20, 25) == "trend"
    assert rsg.detect_market_regime(10, 15, 20) == "range"


def test_calc_period_weights():
    rsg = make_rsg()
    w1, w4, wd = rsg.calc_period_weights(30, 10, 25)
    exp_w1 = 0.6 + 0.4 * min(30, 50) / 50
    exp_w4 = 0.3 + 0.4 * min(10, 50) / 50
    exp_wd = 0.1 + 0.4 * min(25, 50) / 50
    total = exp_w1 + exp_w4 + exp_wd
    assert w1 == pytest.approx(exp_w1 / total)
    assert w4 == pytest.approx(exp_w4 / total)
    assert wd == pytest.approx(exp_wd / total)
    assert pytest.approx(w1 + w4 + wd) == 1.0

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from robust_signal_generator import RobustSignalGenerator


def make_rsg():
    return RobustSignalGenerator.__new__(RobustSignalGenerator)


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

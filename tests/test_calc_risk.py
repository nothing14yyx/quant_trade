import numpy as np
import pytest

from quant_trade.risk_manager import RiskManager


def test_calc_risk_basic():
    rm = RiskManager(cap=5.0)
    val = rm.calc_risk(1.0, pred_vol=0.5, oi_change=0.2, quantile=0.5)
    assert val == pytest.approx(np.quantile([1.0, 0.5, 0.2], 0.5))


def test_calc_risk_cap():
    rm = RiskManager(cap=0.5)
    val = rm.calc_risk(2.0, pred_vol=2.0)
    assert val == 0.5

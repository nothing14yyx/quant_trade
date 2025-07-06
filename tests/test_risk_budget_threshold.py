import numpy as np
import pytest

from quant_trade.robust_signal_generator import risk_budget_threshold


def test_risk_budget_threshold_basic():
    data = [0.1, 0.2, 0.15, 0.4, 0.3]
    th = risk_budget_threshold(data, quantile=0.5)
    assert th == pytest.approx(np.quantile(np.abs(data), 0.5))


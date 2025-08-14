import pytest

from quant_trade.slippage_model import SlippageModel


def test_slippage_bp_negative_hv():
    model = SlippageModel()
    assert model.slippage_bp(-0.5) == 0.0


def test_slippage_bp_extreme_hv():
    model = SlippageModel(hv_cap=0.3)
    assert model.slippage_bp(10.0) == pytest.approx(0.3 * model.bp_per_hv)

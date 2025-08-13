import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.signal.position_sizing import (
    calc_position_size,
    compute_exit_multiplier,
    compute_tp_sl,
)


def test_position_size_and_tp_sl_basic():
    rsg = make_dummy_rsg()
    strength = 0.8
    target = 0.2
    base = calc_position_size(strength, target)
    assert base == pytest.approx(min(abs(strength), target))

    rsg._exit_lag = 0
    rsg.exit_lag_bars = 2
    mult = compute_exit_multiplier(rsg, 1.0, 2.0, 1)
    size = base * mult
    assert size == pytest.approx(base * 0.5)

    tp, sl = compute_tp_sl(rsg, 100, 10, 1)
    assert tp > 100
    assert sl < 100

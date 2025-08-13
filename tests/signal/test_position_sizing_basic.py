import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.signal.position_sizing import (
    calc_position_size,
    compute_exit_multiplier,
    compute_tp_sl,
)
from quant_trade.signal.utils import sigmoid_dir


def test_position_size_and_tp_sl_basic():
    rsg = make_dummy_rsg()
    fused_score = 0.8
    base_th = 0.2
    max_pos = 0.2
    base = calc_position_size(
        fused_score,
        base_th,
        max_position=max_pos,
        gamma=0.5,
    )
    strength = sigmoid_dir(fused_score, base_th, 0.5)
    assert base == pytest.approx(abs(strength) * max_pos)

    weak_score = 0.25
    base2 = calc_position_size(
        weak_score,
        base_th,
        max_position=max_pos,
        gamma=0.5,
        min_exposure=0.2 * max_pos,
    )
    strength_weak = sigmoid_dir(weak_score, base_th, 0.5)
    target_risk = abs(strength_weak) * max_pos
    assert base2 == pytest.approx(max(target_risk, 0.2 * max_pos))

    rsg._exit_lag = 0
    rsg.exit_lag_bars = 2
    mult = compute_exit_multiplier(rsg, 1.0, 2.0, 1)
    size = base * mult
    assert size == pytest.approx(base * 0.5)

    tp, sl = compute_tp_sl(rsg, 100, 10, 1)
    assert tp > 100
    assert sl < 100

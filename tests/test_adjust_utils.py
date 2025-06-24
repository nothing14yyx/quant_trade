import pytest

from quant_trade.robust_signal_generator import (
    adjust_score,
    volume_guard,
    cap_positive,
    fused_to_risk,
)


def test_adjust_score_basic():
    assert adjust_score(+0.6, -0.8) < 0.6
    assert adjust_score(-0.6, -0.8) < -0.6


def test_volume_guard_basic():
    assert abs(volume_guard(+0.5, 0.6, -50)) < 0.5


def test_cap_positive_and_fused_to_risk():
    assert cap_positive(+0.6, -0.6, 0.4) == pytest.approx(+0.24)
    assert fused_to_risk(0.8, 0.02, 0.01, cap=5.0) == 5.0


def test_adjust_score_with_custom_scale():
    assert adjust_score(+0.6, -0.6, cap_scale=0.4) == pytest.approx(+0.168)

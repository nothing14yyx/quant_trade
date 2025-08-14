import pytest

from quant_trade.signal import Vote, fuse_votes


@pytest.mark.parametrize("direction", [1, -1])
def test_all_periods_align_returns_direction(direction: int) -> None:
    v1h = Vote(direction, 0.7)
    v4h = Vote(direction, 0.6)
    vd1 = Vote(direction, 0.9)
    assert fuse_votes(v1h, v4h, vd1) == direction


def test_veto_by_4h_returns_zero() -> None:
    v1h = Vote(1, 0.7)
    v4h = Vote(-1, 0.9)  # probability gap > 0.15 triggers veto
    vd1 = Vote(1, 0.4)
    assert fuse_votes(v1h, v4h, vd1) == 0


def test_veto_by_d1_returns_zero() -> None:
    v1h = Vote(1, 0.7)
    v4h = Vote(1, 0.6)
    vd1 = Vote(-1, 0.7)  # high daily probability triggers veto
    assert fuse_votes(v1h, v4h, vd1) == 0


def test_weighted_align_but_low_prob_returns_zero() -> None:
    v1h = Vote(1, 0.55)
    v4h = Vote(-1, 0.45)
    vd1 = Vote(-1, 0.55)
    assert fuse_votes(v1h, v4h, vd1) == 0

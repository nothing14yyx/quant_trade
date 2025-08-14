import pytest

from quant_trade.signal import Vote, fuse_votes


def test_basic_positive() -> None:
    v1h = Vote(1, 0.6)
    v4h = Vote(1, 0.7)
    vd1 = Vote(1, 0.8)
    assert fuse_votes(v1h, v4h, vd1) == 1


def test_mismatch_weight_returns_zero() -> None:
    v1h = Vote(1, 0.8)
    v4h = Vote(-1, 0.7)
    vd1 = Vote(-1, 0.7)
    assert fuse_votes(v1h, v4h, vd1, w=(1.0, 1.0, 1.0)) == 0


def test_veto_4h_against_1h() -> None:
    v1h = Vote(1, 0.7)
    v4h = Vote(-1, 0.9)
    vd1 = Vote(1, 0.4)
    assert fuse_votes(v1h, v4h, vd1) == 0


def test_veto_d1_against_1h() -> None:
    v1h = Vote(1, 0.7)
    v4h = Vote(1, 0.6)
    vd1 = Vote(-1, 0.7)
    assert fuse_votes(v1h, v4h, vd1) == 0


def test_disable_veto() -> None:
    v1h = Vote(1, 0.7)
    v4h = Vote(-1, 0.9)
    vd1 = Vote(1, 0.4)
    assert fuse_votes(v1h, v4h, vd1, veto=False) == 1


def test_low_prob_blocks_signal() -> None:
    v1h = Vote(1, 0.5)
    v4h = Vote(1, 0.7)
    vd1 = Vote(1, 0.6)
    assert fuse_votes(v1h, v4h, vd1) == 0

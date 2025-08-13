import pytest

from quant_trade.signal.multi_period_fusion import fuse_scores


def test_resonance_amplifies_score():
    scores = {"1h": 0.4, "4h": 0.4, "d1": 0.4}
    weights = (0.5, 0.3, 0.2)
    fused, c_all, c_14, c_4d1 = fuse_scores(scores, weights, False)
    assert c_all
    assert fused > 0.4  # 放大


def test_conflict_penalty():
    scores = {"1h": 0.5, "4h": -0.5, "d1": 0.2}
    weights = (0.5, 0.3, 0.2)
    fused, c_all, c_14, c_4d1 = fuse_scores(scores, weights, False)
    assert not (c_all or c_14 or c_4d1)
    assert fused == pytest.approx(0.5 * 0.7)

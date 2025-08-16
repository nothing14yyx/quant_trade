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


def test_dynamic_ic_adjusts_weights():
    scores = {"1h": 0.5, "4h": 0.2, "d1": 0.1}
    weights = (0.4, 0.4, 0.2)
    ic_stats = {"1h": 0.0, "4h": 1.0, "d1": -0.5}
    fused, c_all, _, _ = fuse_scores(scores, weights, False, ic_stats=ic_stats)
    assert c_all
    # weights after adjustment: 0.4,0.8,0.1 -> normalized to ~0.3077,0.6154,0.0769
    expected = 0.3077 * 0.5 + 0.6154 * 0.2 + 0.0769 * 0.1
    expected *= 1.1
    assert fused == pytest.approx(expected, rel=1e-3)


def test_ignore_period_with_zero_weight():
    scores = {"1h": 0.5, "4h": 0.5, "d1": -0.9}
    weights = (0.5, 0.5, 0.0)
    fused, c_all, c_14, c_4d1 = fuse_scores(scores, weights, False)
    assert c_all and not c_14 and not c_4d1
    assert fused == pytest.approx(0.55)

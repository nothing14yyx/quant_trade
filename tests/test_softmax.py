import numpy as np
import pytest

from quant_trade.robust_signal_generator import softmax


def test_softmax_all_nan():
    res = softmax([float('nan')] * 3)
    assert len(res) == 3
    assert np.all(np.isnan(res))


def test_softmax_mixed_nan():
    arr = [1.0, float('nan'), 2.0, 3.0]
    res = softmax(arr)
    assert np.isnan(res[1])
    expected = np.exp(np.array([1.0, 2.0, 3.0]) - 3.0)
    expected = expected / expected.sum()
    np.testing.assert_allclose(res[[0, 2, 3]], expected)

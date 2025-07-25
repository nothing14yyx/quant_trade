import numpy as np
import pytest

from quant_trade.robust_signal_generator import softmax


def test_softmax_all_nan():
    with pytest.warns(None) as record:
        res = softmax([float('nan')] * 3)
    assert len(res) == 3
    assert not record

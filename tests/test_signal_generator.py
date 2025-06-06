import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from collections import deque

from robust_signal_generator import RobustSignalGenerator


def make_dummy_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.history_scores = deque(maxlen=500)
    rsg.max_same_direction_rate = 0.6
    return rsg


def test_compute_tp_sl():
    rsg = make_dummy_rsg()
    tp, sl = rsg.compute_tp_sl(100, 10, 1)
    assert tp == pytest.approx(115)
    assert sl == pytest.approx(90)

    tp, sl = rsg.compute_tp_sl(100, 10, -1)
    assert tp == pytest.approx(85)
    assert sl == pytest.approx(110)


def test_dynamic_threshold_basic():
    rsg = make_dummy_rsg()
    th = rsg.dynamic_threshold(0, 0, 0)
    assert th == pytest.approx(0.12)


def test_dynamic_threshold_upper_bound():
    rsg = make_dummy_rsg()
    th = rsg.dynamic_threshold(0.1, 50, 0.02)
    assert th == pytest.approx(0.25)


def test_consensus_check():
    rsg = make_dummy_rsg()
    assert rsg.consensus_check(0.2, 0.3, 0.1) == 1
    assert rsg.consensus_check(-0.2, -0.3, 0) == -1
    assert rsg.consensus_check(0.2, -0.3, 0) == 0


def test_crowding_protection():
    rsg = make_dummy_rsg()
    assert rsg.crowding_protection([1, 1, 1, 0, -1]) == 0
    assert rsg.crowding_protection([1, 1, -1, -1]) == 0

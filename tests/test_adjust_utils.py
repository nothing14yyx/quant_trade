import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from robust_signal_generator import adjust_score, volume_guard


def test_adjust_score_basic():
    assert adjust_score(+0.6, -0.8) < 0.6
    assert adjust_score(-0.6, -0.8) < -0.6


def test_volume_guard_basic():
    assert abs(volume_guard(+0.5, 0.6, -50)) < 0.5

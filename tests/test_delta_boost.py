import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from robust_signal_generator import RobustSignalGenerator


def _fake_feat(**kv):
    return kv


def test_delta_boost():
    g = RobustSignalGenerator({}, feature_cols_1h=[], feature_cols_4h=[], feature_cols_d1=[])
    f1 = _fake_feat(rsi_1h=50, macd_hist_1h=0.001)
    f2 = _fake_feat(rsi_1h=57, macd_hist_1h=0.003)
    g._prev_raw["1h"] = f1
    d = g._calc_deltas(f2, f1, g.core_keys["1h"])
    assert d["rsi_1h_delta"] != 0
    s0 = 0.5
    s1 = g._apply_delta_boost(s0, d)
    assert s1 > s0

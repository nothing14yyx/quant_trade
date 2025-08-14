import numpy as np

from quant_trade.utils.lru import LRU
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal.features_to_scores import get_factor_scores_batch


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = LRU(300)
    rsg._ai_score_cache = LRU(300)
    rsg.get_feat_value = lambda row, key, default=0: row.get(key, default)
    rsg._make_cache_key = lambda features, period: (period, tuple(sorted(features.items())))
    return rsg


def test_batch_list_mapping():
    rsg = make_rsg()
    feats = [
        {"rsi_1h": 40, "macd_hist_1h": 0},
        {"rsi_1h": 60, "macd_hist_1h": 0.02},
    ]
    scores = get_factor_scores_batch(rsg, feats, "1h")
    assert scores[1]["momentum"] > scores[0]["momentum"]


def test_batch_structured_array():
    rsg = make_rsg()
    arr = np.array(
        [(40, 0.0), (60, 0.02)],
        dtype=[("rsi_1h", "f4"), ("macd_hist_1h", "f4")],
    )
    scores = get_factor_scores_batch(rsg, arr, "1h")
    assert scores[1]["momentum"] > scores[0]["momentum"]


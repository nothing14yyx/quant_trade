from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal import features_to_scores
from quant_trade.utils.lru import LRU


def test_rsg_generate_signal_batch_calls_vectorized(monkeypatch):
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = LRU(300)
    rsg._ai_score_cache = LRU(300)

    called = {"1h": 0, "4h": 0, "d1": 0}

    def fake_batch(core, feats, period):
        called[period] += 1
        return [{} for _ in feats]

    monkeypatch.setattr(features_to_scores, "get_factor_scores_batch", fake_batch)

    def stub_generate_signal(f1, f4, fd, *a, **k):
        return {}

    rsg.generate_signal = stub_generate_signal

    feats = [{}, {}]
    rsg.generate_signal_batch(feats, feats, feats)

    assert called["1h"] == 1 and called["4h"] == 1 and called["d1"] == 1


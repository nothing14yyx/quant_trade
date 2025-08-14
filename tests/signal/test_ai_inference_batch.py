import threading
from collections.abc import Mapping

import numpy as np

from quant_trade.ai_model_predictor import AIModelPredictor
from quant_trade.signal.ai_inference import compute_ai_scores_batch
from quant_trade.utils.lru import LRU


class ConstPipe:
    def __init__(self, probs, classes=None):
        self._probs = np.array([probs], dtype=float)
        self.classes_ = np.array(classes) if classes is not None else np.arange(len(probs))

    def predict_proba(self, df):
        return np.tile(self._probs, (len(df), 1))


def test_ai_model_predictor_batch():
    model_up = {"pipeline": ConstPipe([0.2, 0.8]), "features": ["a"]}
    model_down = {"pipeline": ConstPipe([0.8, 0.2]), "features": ["a"]}
    predictor = AIModelPredictor.__new__(AIModelPredictor)
    res = predictor.get_ai_score(np.array([[1], [2]]), model_up, model_down)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert np.allclose(res, 0.6)

    model_cls = {"pipeline": ConstPipe([0.2, 0.2, 0.6], classes=[-1, 0, 1]), "features": ["a"]}
    res_cls = predictor.get_ai_score_cls(np.array([[1], [2]]), model_cls)
    assert isinstance(res_cls, np.ndarray)
    assert np.allclose(res_cls, 0.5)


def test_compute_ai_scores_batch_thread_safe():
    class DummyPredictor:
        def get_ai_score(self, feats, *_, **__):
            if isinstance(feats, list) and feats and isinstance(feats[0], Mapping):
                return np.array([f.get("v", 0.0) for f in feats], dtype=float)
            arr = np.asarray(feats, dtype=float)
            if arr.ndim == 1:
                return arr
            return arr[:, 0]

    predictor = DummyPredictor()
    feats = {"1h": [{"v": i} for i in range(5)]}
    models = {"1h": {"up": {}, "down": {}}}
    calibrators = {"1h": {"up": None, "down": None}}
    cache = LRU(maxsize=100)

    res = compute_ai_scores_batch(predictor, feats, models, calibrators, cache)
    assert [r["1h"] for r in res] == [0, 1, 2, 3, 4]

    res_arr = compute_ai_scores_batch(predictor, np.array([[5], [6]]), models, calibrators)
    assert [r["1h"] for r in res_arr] == [5, 6]

    results = []
    def worker():
        results.append(compute_ai_scores_batch(predictor, feats, models, calibrators, cache))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results[0] == res
    assert results[1] == res

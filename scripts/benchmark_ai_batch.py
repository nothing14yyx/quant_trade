import time
import numpy as np

from quant_trade.signal.ai_inference import compute_ai_scores_batch


class DummyPredictor:
    """简单的预测器, 返回常数分数."""

    def get_ai_score(self, feats, *args, **kwargs):
        arr = np.asarray(feats)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.full(arr.shape[0], 0.1)


def main():
    n = 1000
    feats = {"1h": [{"f1": float(i)} for i in range(n)]}
    models = {"1h": {"up": {}, "down": {}}}
    calibrators = {"1h": {"up": None, "down": None}}
    predictor = DummyPredictor()

    start_cpu = time.process_time()
    t0 = time.perf_counter()
    compute_ai_scores_batch(predictor, feats, models, calibrators)
    cpu_time = time.process_time() - start_cpu
    latency = time.perf_counter() - t0
    print(f"CPU time: {cpu_time:.4f}s, latency: {latency:.4f}s")


if __name__ == "__main__":
    main()

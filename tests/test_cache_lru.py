import threading

from quant_trade.utils.lru import LRU
from quant_trade.signal.ai_inference import get_period_ai_scores
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.risk_manager import RiskManager


class DummyPredictor:
    def __init__(self):
        self.calls = 0

    def get_ai_score(self, *args, **kwargs):
        self.calls += 1
        return 0.1

    # 兼容 cls 模型调用
    get_ai_score_cls = get_ai_score


def test_ai_cache_hit_and_eviction_and_init():
    rsg = RobustSignalGenerator()
    rsg.risk_manager = RiskManager()
    assert isinstance(rsg._factor_cache, LRU)
    assert isinstance(rsg._ai_score_cache, LRU)
    assert rsg._factor_cache.maxsize == 300
    assert rsg._ai_score_cache.maxsize == 300

    predictor = DummyPredictor()
    models = {"1h": {"up": {}, "down": {}}}
    feats = {"f": 1}

    # 首次推理应调用预测器
    get_period_ai_scores(predictor, {"1h": feats}, models, cache=rsg._ai_score_cache)
    # 再次调用命中缓存，不增加调用次数
    get_period_ai_scores(predictor, {"1h": feats}, models, cache=rsg._ai_score_cache)
    assert predictor.calls == 1

    # 填满缓存以触发淘汰
    for i in range(2, 302):
        get_period_ai_scores(
            predictor, {"1h": {"f": i}}, models, cache=rsg._ai_score_cache
        )
    assert len(rsg._ai_score_cache) == 300

    prev = predictor.calls
    # 早期条目应被淘汰，重新计算会增加调用次数
    get_period_ai_scores(predictor, {"1h": feats}, models, cache=rsg._ai_score_cache)
    assert predictor.calls == prev + 1


def test_lru_thread_safety():
    cache = LRU(maxsize=50)

    def worker(start):
        for i in range(start, start + 100):
            cache.set(i, i)
            cache.get(i)

    threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cache) <= 50

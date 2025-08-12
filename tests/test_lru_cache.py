import threading

from quant_trade.utils.lru import LRU


def test_set_get_evict():
    cache = LRU(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)  # should evict key "b"
    assert "b" not in cache
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_len_and_contains():
    cache = LRU(maxsize=1)
    assert len(cache) == 0
    cache.set("x", 10)
    assert len(cache) == 1
    assert "x" in cache
    assert cache.get("x") == 10


def test_thread_safety():
    cache = LRU(maxsize=10)

    def writer(key):
        for i in range(100):
            cache.set(key, i)

    threads = [threading.Thread(target=writer, args=(f"k{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cache) <= 10

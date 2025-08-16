from collections import OrderedDict
from threading import Lock
import time


class LRU:
    """线程安全的 LRU 缓存，支持过期清理与命中率统计。"""

    def __init__(self, maxsize: int = 128, ttl: float | None = None) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _purge(self) -> None:
        if self.ttl is None:
            return
        now = time.time()
        keys = [k for k, (_, ts) in self._cache.items() if now - ts > self.ttl]
        for k in keys:
            self._cache.pop(k, None)

    def get(self, key):
        """获取缓存值，若不存在或已过期返回 ``None``。"""
        with self._lock:
            self._purge()
            if key in self._cache:
                value, _ = self._cache[key]
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
                self.hits += 1
                return value
            self.misses += 1
            return None

    def set(self, key, value) -> None:
        """设置缓存值并在超出容量时淘汰最旧的条目。"""
        with self._lock:
            self._purge()
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def cleanup(self) -> None:
        with self._lock:
            self._purge()

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:  # 可选，用于调试和测试
        with self._lock:
            self._purge()
            return len(self._cache)

    def __contains__(self, key) -> bool:  # 可选
        with self._lock:
            self._purge()
            return key in self._cache

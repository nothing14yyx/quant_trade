from collections import OrderedDict
from threading import Lock


class LRU:
    """线程安全的 LRU 缓存。"""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        """获取缓存值，若不存在返回 ``None``。"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key, value) -> None:
        """设置缓存值并在超出容量时淘汰最旧的条目。"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def __len__(self) -> int:  # 可选，用于调试和测试
        with self._lock:
            return len(self._cache)

    def __contains__(self, key) -> bool:  # 可选
        with self._lock:
            return key in self._cache

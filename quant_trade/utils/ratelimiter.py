import threading
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls, period=1.0):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = deque()

    def acquire(self):
        """阻塞直到允许新的调用"""
        while True:
            with self.lock:
                now = time.time()
                while self.calls and self.calls[0] <= now - self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_time = self.calls[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0)

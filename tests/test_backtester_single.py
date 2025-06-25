import pandas as pd
import pytest
import numpy as np
from collections import deque

from quant_trade.param_search import run_single_backtest
from quant_trade.backtester import (
    FEATURE_COLS_1H,
    FEATURE_COLS_4H,
    FEATURE_COLS_D1,
)


class DummySG:
    def __init__(self, signals):
        self.signals = signals
        self.idx = 0
        self.ic_scores = {}
        self.history_scores = deque(maxlen=10)
        self.base_weights = {}

    def generate_signal(self, f1, f4, fd):
        sig = self.signals[self.idx]
        self.idx += 1
        return {"signal": sig, "score": 0.0, "position_size": 1.0}


def test_run_single_backtest_basic():
    times = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "symbol": ["BTC"] * 5,
            "open_time": times,
            "close_time": times + pd.Timedelta(hours=1),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
        }
    )

    for col in FEATURE_COLS_1H:
        df[col] = 0.0
    for col in FEATURE_COLS_4H:
        df[col] = 0.0
    for col in FEATURE_COLS_D1:
        df[col] = 0.0

    sg = DummySG([1, -1, 1, -1])

    avg_ret, avg_sharpe = run_single_backtest(
        df=df,
        base_weights={},
        history_window=10,
        th_params={},
        ic_scores={},
        sg=sg,
    )

    assert avg_ret == pytest.approx(0.0280853, rel=1e-4)
    assert np.isnan(avg_sharpe)


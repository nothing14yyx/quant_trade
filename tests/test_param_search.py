import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import types
import param_search


def dummy_study(*args, **kwargs):
    class DummyTrial:
        def suggest_float(self, name, low, high):
            return low

        def suggest_int(self, name, low, high):
            return low

    class DummyStudy:
        def __init__(self):
            self.best_params = {}
            self.best_value = 0.0

        def optimize(self, func, n_trials, show_progress_bar=False):
            for _ in range(n_trials):
                self.best_value = func(DummyTrial())

    return DummyStudy()

def test_param_search_grid(monkeypatch):
    monkeypatch.setattr(param_search, "load_config", lambda: {})
    monkeypatch.setattr(param_search, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(param_search, "run_single_backtest", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr(param_search, "ParameterGrid", lambda pg: [pg])

    class DummyRSG:
        def __init__(self, *a, **k):
            self.delta_params = {
                "rsi": (1, 1, 0.03),
                "macd_hist": (1, 1, 0.03),
                "ema_diff": (1, 1, 0.03),
                "atr_pct": (1, 1, 0.03),
                "vol_ma_ratio": (1, 1, 0.03),
                "funding_rate": (1, 1, 0.03),
            }
    monkeypatch.setattr(param_search, "RobustSignalGenerator", DummyRSG)

    df = pd.DataFrame({'symbol': [], 'open_time': [], 'close_time': []})
    monkeypatch.setattr(param_search.pd, "read_sql", lambda *a, **k: df)

    param_search.run_param_search(method="grid", trials=1, tune_delta=True)


def test_param_search_optuna(monkeypatch):
    monkeypatch.setattr(param_search, "load_config", lambda: {})
    monkeypatch.setattr(param_search, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(param_search, "run_single_backtest", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr(param_search.optuna, "create_study", lambda direction="maximize": dummy_study())

    class DummyRSG:
        def __init__(self, *a, **k):
            self.delta_params = {
                "rsi": (1, 1, 0.03),
                "macd_hist": (1, 1, 0.03),
                "ema_diff": (1, 1, 0.03),
                "atr_pct": (1, 1, 0.03),
                "vol_ma_ratio": (1, 1, 0.03),
                "funding_rate": (1, 1, 0.03),
            }
    monkeypatch.setattr(param_search, "RobustSignalGenerator", DummyRSG)

    df = pd.DataFrame({'symbol': [], 'open_time': [], 'close_time': []})
    monkeypatch.setattr(param_search.pd, "read_sql", lambda *a, **k: df)

    param_search.run_param_search(method="optuna", trials=1, tune_delta=True)

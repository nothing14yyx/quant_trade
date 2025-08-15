import pandas as pd
import types
import numpy as np
import pytest

from quant_trade import param_search
import quant_trade.utils.db as db


def test_compute_ic_scores_missing_components(caplog):
    class DummyRSG:
        def __init__(self):
            self.base_weights = {"ai": 1}

    df = pd.DataFrame({"open_time": [0], "open": [1], "close": [1]})
    with caplog.at_level("WARNING"):
        result = param_search.compute_ic_scores(df, DummyRSG())
    assert result == {}
    assert "缺少依赖" in caplog.text


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
    monkeypatch.setattr(db, "load_config", lambda: {})
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "load_config", db.load_config)
    monkeypatch.setattr(param_search, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(param_search, "run_single_backtest", lambda *a, **k: (0.0, 0.0, 0))
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

    with pytest.raises(ValueError, match="features 表无数据"):
        param_search.run_param_search(method="grid", trials=1, tune_delta=True)


def test_param_search_optuna(monkeypatch):
    monkeypatch.setattr(db, "load_config", lambda: {})
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "load_config", db.load_config)
    monkeypatch.setattr(param_search, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(param_search, "run_single_backtest", lambda *a, **k: (0.0, 0.0, 0))
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

    with pytest.raises(ValueError, match="features 表无数据"):
        param_search.run_param_search(method="optuna", trials=1, tune_delta=True)


def test_param_search_success(monkeypatch, caplog):
    monkeypatch.setattr(db, "load_config", lambda: {})
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "load_config", db.load_config)
    monkeypatch.setattr(param_search, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(param_search, "ParameterGrid", lambda pg: [pg])

    class DummyRSG:
        def __init__(self, *a, **k):
            self.delta_params = {"rsi": (1, 1, 0.03)}

    monkeypatch.setattr(param_search, "RobustSignalGenerator", DummyRSG)

    times = pd.date_range("2020-01-01", periods=3, freq="h")
    df = pd.DataFrame({
        "symbol": ["BTC"] * 3,
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=1),
    })
    monkeypatch.setattr(param_search.pd, "read_sql", lambda *a, **k: df)

    monkeypatch.setattr(param_search, "run_single_backtest", lambda *a, **k: (0.1, 0.2, 0))

    import quant_trade.backtester as backtester

    monkeypatch.setattr(backtester, "simulate_trades", lambda *a, **k: pd.DataFrame())

    with caplog.at_level("INFO"):
        with pytest.raises(ValueError, match="no trades found during parameter search"):
            param_search.run_param_search(method="grid", trials=1, tune_delta=False)


def test_param_search_cv(monkeypatch):
    monkeypatch.setattr(db, "load_config", lambda: {})
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "load_config", db.load_config)
    monkeypatch.setattr(param_search, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(param_search, "ParameterGrid", lambda pg: [pg])
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})

    class DummyRSG:
        def __init__(self, *a, **k):
            self.delta_params = {"rsi": (1, 1, 0.03)}

    monkeypatch.setattr(param_search, "RobustSignalGenerator", DummyRSG)

    times = pd.date_range("2020-01-01", periods=4, freq="h")
    df = pd.DataFrame({
        "symbol": ["BTC"] * 4,
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=1),
    })
    monkeypatch.setattr(param_search.pd, "read_sql", lambda *a, **k: df)

    calls = []
    sharpe_vals = [0.1, 0.3]

    def fake_backtest(*args, **kwargs):
        idx = len(calls)
        calls.append(1)
        return 0.0, sharpe_vals[idx], 0

    monkeypatch.setattr(param_search, "run_single_backtest", fake_backtest)

    with pytest.raises(ValueError, match="no trades found during parameter search"):
        param_search.run_param_search(method="grid", tune_delta=False, n_splits=2)

    assert len(calls) == 2


def test_param_search_nan_metric(monkeypatch):
    monkeypatch.setattr(db, "load_config", lambda: {})
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(param_search, "load_config", db.load_config)
    monkeypatch.setattr(param_search, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(param_search, "ParameterGrid", lambda pg: [pg])
    monkeypatch.setattr(param_search, "precompute_ic_scores", lambda df, sg: {})

    class DummyRSG:
        def __init__(self, *a, **k):
            self.delta_params = {"rsi": (1, 1, 0.03)}

    monkeypatch.setattr(param_search, "RobustSignalGenerator", DummyRSG)

    times = pd.date_range("2020-01-01", periods=2, freq="h")
    df = pd.DataFrame({
        "symbol": ["BTC", "BTC"],
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=1),
    })
    monkeypatch.setattr(param_search.pd, "read_sql", lambda *a, **k: df)

    monkeypatch.setattr(
        param_search,
        "run_single_backtest",
        lambda *a, **k: (np.nan, np.nan, 0),
    )

    with pytest.raises(ValueError, match="no trades found during parameter search"):
        param_search.run_param_search(method="grid", tune_delta=False)

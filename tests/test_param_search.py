import pandas as pd
import types
import pytest

from quant_trade import param_search


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
    monkeypatch.setattr(param_search, "load_config", lambda: {})
    monkeypatch.setattr(param_search, "connect_mysql", lambda cfg: None)
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
    monkeypatch.setattr(param_search, "load_config", lambda: {})
    monkeypatch.setattr(param_search, "connect_mysql", lambda cfg: None)
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
        param_search.run_param_search(method="grid", trials=1, tune_delta=False)

    assert any("best params:" in record.getMessage() for record in caplog.records)

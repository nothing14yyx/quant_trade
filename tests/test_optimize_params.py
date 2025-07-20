import types
import pandas as pd
import yaml
import pytest

import optimize_params as op


def dummy_study(*args, **kwargs):
    class DummyTrial:
        def suggest_float(self, name, low, high):
            return low

    class DummyStudy:
        def __init__(self):
            self.best_params = {}
            self.best_value = 0.0

        def optimize(self, func, n_trials, show_progress_bar=False):
            for _ in range(n_trials):
                self.best_params = {
                    "base_th": 0.05,
                    "risk_adjust_factor": 0.1,
                    "ai_w": 0.1,
                    "trend_w": 0.1,
                    "momentum_w": 0.1,
                    "volatility_w": 0.1,
                    "volume_w": 0.1,
                    "sentiment_w": 0.1,
                    "funding_w": 0.1,
                }
                self.best_value = func(DummyTrial())

    return DummyStudy()


def test_optimize_params(monkeypatch, tmp_path):
    monkeypatch.setattr(op, "load_config", lambda: {
        "mysql": {"host": "", "user": "", "password": "", "database": ""},
        "signal_threshold": {"base_th": 0.1},
        "ic_scores": {"base_weights": {k: 1/7 for k in op.KEYS}},
        "risk_adjust": {"factor": 0.3},
    })
    monkeypatch.setattr(op, "connect_mysql", lambda cfg: None)

    df = pd.DataFrame({
        "symbol": ["BTC"],
        "open_time": pd.date_range("2020-01-01", periods=1, freq="h"),
        "close_time": pd.date_range("2020-01-01", periods=1, freq="h"),
    })
    monkeypatch.setattr(op.pd, "read_sql", lambda *a, **k: df)

    class DummyRSG:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(op, "RobustSignalGenerator", DummyRSG)
    monkeypatch.setattr(op, "precompute_ic_scores", lambda df, sg: {})
    monkeypatch.setattr(op, "run_single_backtest", lambda *a, **k: (0.1, 0.2, 1))
    monkeypatch.setattr(op.optuna, "create_study", lambda direction="maximize": dummy_study())

    out_file = tmp_path / "config.yaml"
    best = op.optimize_params(rows=1, trials=1, out_path=out_file)
    assert isinstance(best, dict)
    data = yaml.safe_load(out_file.read_text())
    assert data["signal_threshold"]["base_th"] == best["base_th"]

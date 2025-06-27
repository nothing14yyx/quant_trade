import pandas as pd

import quant_trade.generate_signal_from_db as gsdb


class DummySG:
    def __init__(self, *a, **k):
        pass

    def set_symbol_categories(self, categories):
        pass

    def generate_signal(self, *a, **k):
        return None


def test_main_handles_none(monkeypatch, capsys):
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=1, freq="h"),
        "close": [1.0],
    })

    monkeypatch.setattr(gsdb, "load_config", lambda path=gsdb.CONFIG_PATH: {
        "mysql": {},
        "feature_engineering": {"scaler_path": ""},
        "models": {},
    })
    monkeypatch.setattr(gsdb, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(gsdb, "load_latest_klines", lambda *a, **k: df)
    monkeypatch.setattr(gsdb, "load_scaler_params_from_json", lambda p: {})
    monkeypatch.setattr(gsdb, "load_global_metrics", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "load_latest_open_interest", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "load_order_book_imbalance", lambda *a, **k: None)
    monkeypatch.setattr(gsdb, "load_symbol_categories", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "prepare_all_features", lambda *a, **k: ({}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(gsdb, "RobustSignalGenerator", lambda *a, **k: DummySG())
    monkeypatch.setattr(gsdb, "collect_feature_cols", lambda *a, **k: [])

    gsdb.main("AAA")
    out, _ = capsys.readouterr()
    assert "Empty DataFrame" in out

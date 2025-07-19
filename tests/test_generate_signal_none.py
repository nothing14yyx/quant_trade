import pandas as pd
import logging

import quant_trade.generate_signal_from_db as gsdb
import quant_trade.utils.db as db


class DummySG:
    def __init__(self, *a, **k):
        pass

    def set_symbol_categories(self, categories):
        pass

    def generate_signal(self, *a, **k):
        return None


def test_main_handles_none(monkeypatch, caplog):
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=1, freq="h"),
        "close": [1.0],
    })

    monkeypatch.setattr(db, "load_config", lambda path=db.CONFIG_PATH: {
        "mysql": {},
        "feature_engineering": {"scaler_path": ""},
        "models": {},
    })
    monkeypatch.setattr(db, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(gsdb, "load_config", db.load_config)
    monkeypatch.setattr(gsdb, "connect_mysql", db.connect_mysql)
    monkeypatch.setattr(gsdb, "load_latest_klines", lambda *a, **k: df)
    monkeypatch.setattr(gsdb, "load_scaler_params_from_json", lambda p: {})
    monkeypatch.setattr(gsdb, "load_global_metrics", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "load_latest_open_interest", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "load_order_book_imbalance", lambda *a, **k: None)
    monkeypatch.setattr(gsdb, "load_symbol_categories", lambda *a, **k: {})
    monkeypatch.setattr(gsdb, "prepare_all_features", lambda *a, **k: ({}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(gsdb, "RobustSignalGenerator", lambda *a, **k: DummySG())
    monkeypatch.setattr(gsdb, "collect_feature_cols", lambda *a, **k: [])

    with caplog.at_level(logging.INFO):
        gsdb.main("AAA")
    assert "Empty DataFrame" in caplog.text

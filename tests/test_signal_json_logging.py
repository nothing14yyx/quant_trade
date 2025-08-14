import json
import random
import pandas as pd

from quant_trade import json_logger
import quant_trade.backtester as bt


def test_signal_json_logging(monkeypatch, tmp_path):
    # Redirect logs to temporary directory
    monkeypatch.setattr(json_logger, "LOG_DIR", tmp_path / "logs")

    # Minimal configuration and stubs
    monkeypatch.setattr(bt, "load_config", lambda: {})
    monkeypatch.setattr(bt, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(bt, "FEATURE_COLS_1H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_4H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_D1", [])
    monkeypatch.setattr(bt, "calc_features_raw", lambda df, period: pd.DataFrame(index=df.index))
    monkeypatch.setattr(bt, "simulate_trades", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)

    times = pd.date_range("2024-01-01", periods=11, freq="H")
    df = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 11,
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=1),
        "open": [1] * 11,
        "high": [1] * 11,
        "low": [1] * 11,
        "close": [1] * 11,
        "volume": [1] * 11,
    })

    def fake_read_sql(sql, engine, params=None, parse_dates=None):
        if "MAX(open_time" in sql:
            return pd.DataFrame({"end_time": [times[-1]]})
        return df

    monkeypatch.setattr(bt.pd, "read_sql", fake_read_sql)

    class DummyRSG:
        def __init__(self, cfg):
            pass

        def update_ic_scores(self, df, group_by=None):
            pass

        def generate_signal_batch(self, f1, f4, fd):
            results = []
            for i in range(len(f1)):
                results.append(
                    {
                        "signal": 1,
                        "score": float(i),
                        "position_size": 0.5,
                        "take_profit": None,
                        "stop_loss": None,
                        "details": {
                            "penalties": ["p1"],
                            "score_mult": 0.8,
                            "pos_mult": 0.6,
                            "base_th": 0.1,
                        },
                    }
                )
            return results

    monkeypatch.setattr(bt, "RobustSignalGenerator", DummyRSG)

    bt.run_backtest(recent_days=1)

    log_file = json_logger.LOG_DIR / f"signal_{times[1].strftime('%Y%m%d')}.jsonl"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    sample = random.sample(records, 5)
    for rec in sample:
        assert "reasons" in rec
        assert "score_mult" in rec
        assert "pos_mult" in rec
        assert "base_th" in rec

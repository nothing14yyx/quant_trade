import json
import random
import pandas as pd

from quant_trade import json_logger
import quant_trade.backtester as bt


def test_backtest_logs_include_risk_fields(monkeypatch, tmp_path):
    """运行一天回测并验证日志记录包含关键风险字段"""
    # 重定向日志目录
    monkeypatch.setattr(json_logger, "LOG_DIR", tmp_path / "logs")

    # 简化依赖，避免真实数据库和特征计算
    monkeypatch.setattr(bt, "load_config", lambda: {})
    monkeypatch.setattr(bt, "connect_mysql", lambda cfg: None)
    monkeypatch.setattr(bt, "FEATURE_COLS_1H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_4H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_D1", [])
    monkeypatch.setattr(bt, "calc_features_raw", lambda df, period: pd.DataFrame(index=df.index))
    monkeypatch.setattr(bt, "simulate_trades", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)

    # 构造简单的行情数据
    times = pd.date_range("2024-01-01", periods=11, freq="H")
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 11,
            "open_time": times,
            "close_time": times + pd.Timedelta(hours=1),
            "open": [1] * 11,
            "high": [1] * 11,
            "low": [1] * 11,
            "close": [1] * 11,
            "volume": [1] * 11,
        }
    )

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
                            "crowding_factor": 0.9,
                            "oi_threshold": 0.5,
                            "risk_score": 0.2,
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
        assert "base_th" in rec
        details = rec.get("details", {})
        for key in ("crowding_factor", "oi_threshold", "risk_score"):
            assert key in details

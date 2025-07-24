import pandas as pd
import sqlalchemy
import yaml

from quant_trade.market_phase import detect_market_phase


def _setup(engine, data):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp INTEGER, metric TEXT, value REAL)"
        )
    pd.DataFrame(data).to_sql("cm_onchain_metrics", engine, if_exists="append", index=False)


def test_detect_market_phase_multi_chain(tmp_path):
    engine = sqlalchemy.create_engine("sqlite:///:memory:")

    data = []
    for i in range(39):
        for sym in ["BTCUSDT", "ETHUSDT", "LTCUSDT"]:
            for m, v in [
                ("AdrActCnt", 100),
                ("CapMrktCurUSD", 1000),
                ("FeeTotUSD", 1),
            ]:
                data.append({"symbol": sym, "timestamp": i, "metric": m, "value": v})

    data.extend([
        {"symbol": "BTCUSDT", "timestamp": 39, "metric": "AdrActCnt", "value": 200},
        {"symbol": "BTCUSDT", "timestamp": 39, "metric": "CapMrktCurUSD", "value": 2000},
        {"symbol": "BTCUSDT", "timestamp": 39, "metric": "FeeTotUSD", "value": 2},
        {"symbol": "ETHUSDT", "timestamp": 39, "metric": "AdrActCnt", "value": 50},
        {"symbol": "ETHUSDT", "timestamp": 39, "metric": "CapMrktCurUSD", "value": 500},
        {"symbol": "ETHUSDT", "timestamp": 39, "metric": "FeeTotUSD", "value": 0.5},
        {"symbol": "LTCUSDT", "timestamp": 39, "metric": "AdrActCnt", "value": 100},
        {"symbol": "LTCUSDT", "timestamp": 39, "metric": "CapMrktCurUSD", "value": 1000},
        {"symbol": "LTCUSDT", "timestamp": 39, "metric": "FeeTotUSD", "value": 1},
    ])

    _setup(engine, data)

    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump({"market_phase": {"symbols": ["BTCUSDT", "ETHUSDT", "LTCUSDT"]}}, cfg_path.open("w"))

    res = detect_market_phase(engine, cfg_path)
    assert isinstance(res, dict)

    assert res["BTCUSDT"]["phase"] == "bull"
    assert res["ETHUSDT"]["phase"] == "bear"
    assert res["LTCUSDT"]["phase"] == "range"

    assert res["BTCUSDT"]["S"] > 1.5
    assert res["ETHUSDT"]["S"] < 0.8
    assert abs(res["LTCUSDT"]["S"] - 1.0) < 1e-6

import pandas as pd
import sqlalchemy
import yaml
import pytest

from quant_trade.market_phase import detect_market_phase


def _prepare_engine():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp INTEGER, metric TEXT, value REAL)"
        )
    return engine


def _insert_sample_data(engine):
    data = []
    # 30 days baseline
    for i in range(30):
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            for m, v in [
                ("AdrActCnt", 100),
                ("CapMrktCurUSD", 1000),
                ("FeeTotUSD", 1),
            ]:
                data.append({"symbol": sym, "timestamp": i, "metric": m, "value": v})
    # day 30 values decide phase
    for m, v in [("AdrActCnt", 200), ("CapMrktCurUSD", 2000), ("FeeTotUSD", 2)]:
        data.append({"symbol": "BTCUSDT", "timestamp": 30, "metric": m, "value": v})
    for m, v in [("AdrActCnt", 50), ("CapMrktCurUSD", 500), ("FeeTotUSD", 0.5)]:
        data.append({"symbol": "ETHUSDT", "timestamp": 30, "metric": m, "value": v})
    for m, v in [("AdrActCnt", 100), ("CapMrktCurUSD", 1000), ("FeeTotUSD", 1)]:
        data.append({"symbol": "SOLUSDT", "timestamp": 30, "metric": m, "value": v})
    pd.DataFrame(data).to_sql("cm_onchain_metrics", engine, if_exists="append", index=False)


@pytest.fixture()
def engine_with_data():
    engine = _prepare_engine()
    _insert_sample_data(engine)
    return engine


def test_detect_market_phase_multi_chain(engine_with_data, tmp_path):
    cfg = {"market_phase": {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}}
    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump(cfg, cfg_path.open("w"))

    res = detect_market_phase(engine_with_data, cfg_path)

    assert res["BTCUSDT"]["phase"] == "bull" and res["BTCUSDT"]["S"] > 0
    assert res["ETHUSDT"]["phase"] == "bear" and res["ETHUSDT"]["S"] < 0
    assert res["SOLUSDT"]["phase"] == "range" and abs(res["SOLUSDT"]["S"]) < 1e-8

    total_cap = 2000 + 500 + 1000
    expected_total = (
        res["BTCUSDT"]["S"] * 2000 / total_cap
        + res["ETHUSDT"]["S"] * 500 / total_cap
        + res["SOLUSDT"]["S"] * 1000 / total_cap
    )
    assert res["TOTAL"]["phase"] == "bull"
    assert res["TOTAL"]["S"] == pytest.approx(expected_total)
    assert res["window"] == 30

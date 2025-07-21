import pandas as pd
import sqlalchemy

from quant_trade.market_phase import detect_market_phase
from quant_trade.tests.test_utils import make_dummy_rsg


def _setup(engine, data):
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp INTEGER, metric TEXT, value REAL)"
        )
    df = pd.DataFrame(data)
    df.to_sql("cm_onchain_metrics", engine, if_exists="append", index=False)


def test_detect_market_phase():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    data = []
    for i in range(39):
        data.append({"symbol": "BTC", "timestamp": i, "metric": "AdrActCnt", "value": 100})
        data.append({"symbol": "BTC", "timestamp": i, "metric": "Sopr", "value": 0.9})
    data.append({"symbol": "BTC", "timestamp": 39, "metric": "AdrActCnt", "value": 200})
    data.append({"symbol": "BTC", "timestamp": 39, "metric": "Sopr", "value": 1.05})
    _setup(engine, data)
    phase = detect_market_phase(engine)
    assert phase == "bull"


def test_phase_threshold_adjustment():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    data = []
    for i in range(39):
        data.append({"symbol": "BTC", "timestamp": i, "metric": "AdrActCnt", "value": 100})
        data.append({"symbol": "BTC", "timestamp": i, "metric": "Sopr", "value": 0.9})
    data.append({"symbol": "BTC", "timestamp": 39, "metric": "AdrActCnt", "value": 200})
    data.append({"symbol": "BTC", "timestamp": 39, "metric": "Sopr", "value": 1.05})
    _setup(engine, data)

    rsg = make_dummy_rsg()
    rsg.update_market_phase(engine)
    assert rsg.phase_th_mult < 1.0
    res = rsg.apply_risk_filters(
        fused_score=0.2,
        logic_score=0.2,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h={},
        raw_f4h={},
        raw_fd1={},
        vol_preds={},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache={"history_scores": []},
        global_metrics=None,
        symbol=None,
    )
    assert res["base_th"] < rsg.signal_params.base_th


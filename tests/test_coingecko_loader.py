import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import sqlalchemy
import requests
import pytest
from data_loader import DataLoader
from utils.ratelimiter import RateLimiter


def make_dl(engine):
    dl = DataLoader.__new__(DataLoader)
    dl.cg_api_key = ''
    dl.cg_rate_limiter = RateLimiter(30, 60)
    dl._cg_id_map = {}
    dl.retries = 1
    dl.backoff = 0
    dl.engine = engine
    return dl


def test_cg_get_id_cached(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    dl = make_dl(engine)

    calls = []

    def fake_get(url, params=None, headers=None, timeout=10):
        calls.append((url, params))
        class R:
            def json(self):
                return {"coins": [{"id": "bitcoin", "symbol": "btc"}]}
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    cid1 = dl._cg_get_id('BTCUSDT')
    cid2 = dl._cg_get_id('BTCUSDT')

    assert cid1 == 'bitcoin'
    assert cid2 == 'bitcoin'
    assert len(calls) == 1


def test_update_cg_market_data(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cg_market_data (symbol TEXT, timestamp TEXT, price REAL, market_cap REAL, total_volume REAL, PRIMARY KEY(symbol, timestamp))"
        )
    dl = make_dl(engine)

    monkeypatch.setattr(dl, '_cg_get_id', lambda s: 'bitcoin')

    def fake_get(url, params=None, headers=None, timeout=10):
        class R:
            def json(self):
                return {
                    "prices": [[0, 1]],
                    "market_caps": [[0, 10]],
                    "total_volumes": [[0, 5]],
                }
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    dl.update_cg_market_data(['BTCUSDT'])

    df = pd.read_sql('cg_market_data', engine)
    assert len(df) == 1
    assert df.iloc[0]['price'] == 1
    assert df.iloc[0]['market_cap'] == 10
    assert df.iloc[0]['total_volume'] == 5

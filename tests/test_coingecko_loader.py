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


def test_update_cg_global_metrics(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cg_global_metrics (timestamp TEXT PRIMARY KEY, total_market_cap REAL, total_volume REAL, btc_dominance REAL, eth_dominance REAL)"
        )
    dl = make_dl(engine)

    def fake_get(url, headers=None, timeout=10):
        class R:
            def json(self):
                return {
                    "data": {
                        "total_market_cap": {"usd": 100},
                        "total_volume": {"usd": 10},
                        "market_cap_percentage": {"btc": 50, "eth": 30},
                    }
                }

        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    dl.update_cg_global_metrics()

    df = pd.read_sql('cg_global_metrics', engine, parse_dates=['timestamp'])
    assert len(df) == 1
    assert df.iloc[0]['total_market_cap'] == 100
    assert df.iloc[0]['total_volume'] == 10
    assert df.iloc[0]['btc_dominance'] == 50
    assert df.iloc[0]['eth_dominance'] == 30
    ts = df.iloc[0]['timestamp']
    assert ts.tzinfo is None
    assert ts.microsecond % 1000 == 0
    assert ts.minute == 0
    assert ts.second == 0


def test_update_cg_global_metrics_skip(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cg_global_metrics (timestamp TEXT PRIMARY KEY, total_market_cap REAL, total_volume REAL, btc_dominance REAL, eth_dominance REAL)"
        )
    dl = make_dl(engine)

    def fake_get(url, headers=None, timeout=10):
        class R:
            def json(self):
                return {
                    "data": {
                        "total_market_cap": {"usd": 100},
                        "total_volume": {"usd": 10},
                        "market_cap_percentage": {"btc": 50, "eth": 30},
                    }
                }

        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    dl.update_cg_global_metrics()
    dl.update_cg_global_metrics(min_interval_hours=1)

    df = pd.read_sql('cg_global_metrics', engine)
    assert len(df) == 1


def test_update_cg_category_stats(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cg_category_stats (id TEXT PRIMARY KEY, name TEXT, market_cap REAL, market_cap_change_24h REAL, volume_24h REAL, top_3_coins TEXT, updated_at TEXT)"
        )
    dl = make_dl(engine)

    def fake_get(url, headers=None, timeout=10):
        class R:
            def json(self):
                return [{
                    "id": "gamefi",
                    "name": "GameFi",
                    "market_cap": 1000,
                    "market_cap_change_24h": 5.0,
                    "volume_24h": 200,
                    "top_3_coins": ["a", "b", "c"],
                    "updated_at": "2024-06-12T00:00:00Z",
                }]
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    dl.update_cg_category_stats()

    df = pd.read_sql('cg_category_stats', engine)
    assert len(df) == 1
    row = df.iloc[0]
    assert row['id'] == 'gamefi'
    assert row['name'] == 'GameFi'
    assert row['market_cap'] == 1000
    assert row['market_cap_change_24h'] == 5.0
    assert row['volume_24h'] == 200
    assert row['top_3_coins'] == 'a,b,c'

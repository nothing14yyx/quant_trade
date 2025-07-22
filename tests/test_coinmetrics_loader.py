import pandas as pd
import sqlalchemy
import requests
import logging

from quant_trade.coinmetrics_loader import CoinMetricsLoader
from quant_trade.data_loader import DataLoader


def make_loader(engine):
    loader = CoinMetricsLoader.__new__(CoinMetricsLoader)
    loader.engine = engine
    loader.api_key = ''
    loader.metrics = ['AdrActCnt', 'AdrNewCnt', 'SplyCur', 'FeesUSD']
    loader.retries = 1
    loader.backoff = 0
    loader.rate_limiter = type('RL', (), {
        'acquire': lambda self: None,
        'max_calls': 10,
        'period': 6.0,
    })()
    return loader


def test_update_cm_metrics(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))'
        )
    loader = make_loader(engine)

    calls = []

    def fake_get(url, params=None, headers=None, timeout=10):
        calls.append(params['metrics'])
        class R:
            def json(self):
                return {
                    'data': [{
                        'asset': 'btc',
                        'time': '2024-06-01T00:00:00Z',
                        'AdrActCnt': '10',
                        'AdrNewCnt': '5',
                        'SplyCur': '100',
                        'FeesUSD': '2'
                    }]
                }
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    loader.update_cm_metrics(['BTCUSDT'], batch_size=2)

    assert calls == ['AdrActCnt,AdrNewCnt', 'SplyCur,FeesUSD']

    df = pd.read_sql('cm_onchain_metrics', engine)
    assert len(df) == 4
    assert set(df['metric']) == {'AdrActCnt', 'AdrNewCnt', 'SplyCur', 'FeesUSD'}
    assert df['value'].sum() == 117


def make_dl(engine):
    dl = DataLoader.__new__(DataLoader)
    dl.cm_loader = make_loader(engine)
    return dl


def test_data_loader_update_cm_metrics(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))'
        )
    dl = make_dl(engine)

    def fake_get(url, params=None, headers=None, timeout=10):
        class R:
            def json(self):
                return {
                    'data': [{
                        'asset': 'btc',
                        'time': '2024-06-02T00:00:00Z',
                        'AdrActCnt': '1',
                        'AdrNewCnt': '2',
                        'SplyCur': '3'
                    }]
                }
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    dl.update_cm_metrics(['BTCUSDT'])

    df = pd.read_sql('cm_onchain_metrics', engine)
    assert len(df) == 3
    assert set(df['metric']) == {'AdrActCnt', 'AdrNewCnt', 'SplyCur'}
    assert df['value'].sum() == 6


def test_update_cm_metrics_api_error(monkeypatch, caplog):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))'
        )
    loader = make_loader(engine)

    def fake_get(url, params=None, headers=None, timeout=10):
        class R:
            def json(self):
                return {'error_msg': 'invalid metric'}
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    with caplog.at_level(logging.WARNING):
        loader.update_cm_metrics(['BTCUSDT'])

    assert 'invalid metric' in caplog.text
    df = pd.read_sql('cm_onchain_metrics', engine)
    assert df.empty


def test_update_cm_metrics_community_filter(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))'
        )
    loader = make_loader(engine)

    monkeypatch.setattr(loader, 'community_metrics', lambda asset: ['AdrActCnt', 'SplyCur'])

    calls = []

    def fake_get(url, params=None, headers=None, timeout=10):
        calls.append(params['metrics'])
        class R:
            def json(self):
                return {
                    'data': [{
                        'asset': 'btc',
                        'time': '2024-06-03T00:00:00Z',
                        'AdrActCnt': '10',
                        'SplyCur': '100'
                    }]
                }
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    loader.update_cm_metrics(['BTCUSDT'], batch_size=2, community_only=True)

    assert calls == ['AdrActCnt,SplyCur']

    df = pd.read_sql('cm_onchain_metrics', engine)
    assert len(df) == 2
    assert set(df['metric']) == {'AdrActCnt', 'SplyCur'}

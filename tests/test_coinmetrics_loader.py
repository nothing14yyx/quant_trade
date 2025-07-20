import pandas as pd
import sqlalchemy
import requests

from quant_trade.coinmetrics_loader import CoinMetricsLoader


def make_loader(engine):
    loader = CoinMetricsLoader.__new__(CoinMetricsLoader)
    loader.engine = engine
    loader.api_key = ''
    loader.metrics = ['AdrActCnt', 'AdrNewCnt']
    loader.retries = 1
    loader.backoff = 0
    loader.rate_limiter = type('RL', (), {'acquire': lambda self: None})()
    return loader


def test_update_cm_metrics(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            'CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))'
        )
    loader = make_loader(engine)

    def fake_get(url, params=None, headers=None, timeout=10):
        assert params['metrics'] == 'AdrActCnt,AdrNewCnt'
        class R:
            def json(self):
                return {
                    'data': [{
                        'asset': 'btc',
                        'time': '2024-06-01T00:00:00Z',
                        'AdrActCnt': '10',
                        'AdrNewCnt': '5'
                    }]
                }
        return R()

    monkeypatch.setattr(requests, 'get', fake_get)

    loader.update_cm_metrics(['BTCUSDT'])

    df = pd.read_sql('cm_onchain_metrics', engine)
    assert len(df) == 2
    assert set(df['metric']) == {'AdrActCnt', 'AdrNewCnt'}
    assert df['value'].sum() == 15

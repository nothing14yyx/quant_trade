import pandas as pd
import sqlalchemy

from quant_trade.feature_engineering import FeatureEngineer


def test_load_klines_db_cm_metrics():
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE cm_onchain_metrics (symbol TEXT, timestamp TEXT, metric TEXT, value REAL, PRIMARY KEY(symbol, timestamp, metric))"
        )

    times = pd.date_range('2020-01-01', periods=2, freq='h')
    metrics = ['AdrActCnt', 'AdrNewCnt', 'TxCnt', 'CapMrktCurUSD', 'CapRealUSD']
    rows = []
    for t in times:
        for i, m in enumerate(metrics, start=1):
            rows.append({'symbol': 'BTCUSDT', 'timestamp': t.isoformat(), 'metric': m, 'value': i * 10 + t.hour})
    pd.DataFrame(rows).to_sql('cm_onchain_metrics', engine, index=False, if_exists='append')

    kl_df = pd.DataFrame({
        'open': [1, 2],
        'high': [1, 2],
        'low': [1, 2],
        'close': [1, 2],
        'close_time': times + pd.Timedelta(hours=1),
        'volume': [1, 1],
        'quote_asset_volume': [1, 1],
        'num_trades': [1, 1],
        'taker_buy_base': [0.5, 0.5],
        'taker_buy_quote': [0.5, 0.5],
    }, index=times)
    kl_df.index.name = 'open_time'

    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.engine = engine
    fe.btc_symbol = 'BTCUSDT'
    fe.eth_symbol = 'ETHUSDT'
    fe.cm_metrics = metrics
    fe._kl_cache = {}
    fe._load_klines_raw = lambda symbol, interval: kl_df

    res = fe.load_klines_db('BTCUSDT', '1h')
    assert 'AdrActCnt' in res.columns
    assert 'CapRealUSD' in res.columns
    assert res['AdrActCnt'].iloc[0] == 10
    assert res['CapRealUSD'].iloc[1] == 51

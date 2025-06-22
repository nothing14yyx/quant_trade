import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from utils.helper import calc_order_book_features
from feature_engineering import FeatureEngineer
from data_loader import DataLoader
import sqlalchemy


def test_calc_order_book_features():
    times = pd.date_range('2020-01-01', periods=3, freq='h')
    bids = json.dumps([[1, 2]] * 10)
    asks = json.dumps([[1, 1]] * 10)
    df = pd.DataFrame({
        'timestamp': times,
        'bids': [bids]*3,
        'asks': [asks]*3,
    })
    feats = calc_order_book_features(df)
    assert 'bid_ask_imbalance' in feats
    expected = (20 - 10) / 30
    assert feats['bid_ask_imbalance'].iloc[0] == pytest.approx(expected)


def test_merge_features_with_order_book(tmp_path):
    fe = FeatureEngineer()
    fe.feature_cols_path = tmp_path / 'cols.txt'

    times1h = pd.date_range('2020-01-01', periods=60, freq='h')
    df1h = pd.DataFrame({
        'open':1,'high':1,'low':1,'close':1,'volume':1,
        'close_time': times1h + pd.Timedelta(hours=1),
        'quote_asset_volume':1,
        'num_trades':1,
        'taker_buy_base':0.5,
        'taker_buy_quote':0.5,
    }, index=times1h)
    df1h.index.name = 'open_time'
    times4h = pd.date_range('2020-01-01', periods=50, freq='4h')
    df4h = pd.DataFrame({
        'open':1,'high':1,'low':1,'close':1,'volume':1,
        'close_time': times4h + pd.Timedelta(hours=4),
        'quote_asset_volume':1,
        'num_trades':1,
        'taker_buy_base':0.5,
        'taker_buy_quote':0.5,
    }, index=times4h)
    df4h.index.name = 'open_time'
    times1d = pd.date_range('2020-01-01', periods=50, freq='D')
    df1d = pd.DataFrame({
        'open':1,'high':1,'low':1,'close':1,'volume':1,
        'close_time': times1d + pd.Timedelta(days=1),
        'quote_asset_volume':1,
        'num_trades':1,
        'taker_buy_base':0.5,
        'taker_buy_quote':0.5,
    }, index=times1d)
    df1d.index.name = 'open_time'
    bids = json.dumps([[1,2]]*10)
    asks = json.dumps([[1,1]]*10)
    ob_df = pd.DataFrame({'timestamp': times1h, 'bids':[bids]*60, 'asks':[asks]*60})

    fe.load_klines_db = lambda sym, iv: {'1h': df1h, '4h': df4h, 'd1': df1d}[iv]
    fe.load_order_book = lambda sym: ob_df
    fe.get_symbols = lambda intervals=("1h","4h","d1"): ['BTC']
    fe._write_output = lambda df, save_to_db, append: setattr(fe, 'out_df', df)

    fe.merge_features(save_to_db=False, batch_size=None)
    out_df = fe.out_df
    assert 'bid_ask_imbalance' in out_df.columns
    assert out_df['bid_ask_imbalance'].iloc[0] == pytest.approx((20-10)/30)


def test_get_latest_order_book_imbalance():
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE order_book (symbol TEXT, timestamp TEXT, bids TEXT, asks TEXT, PRIMARY KEY(symbol, timestamp))"
        )
    dl = DataLoader.__new__(DataLoader)
    dl.engine = engine
    dl.retries = 1
    dl.backoff = 0

    ts = pd.Timestamp('2020-01-01 00:00:00')
    bids = json.dumps([[1, 2]] * 10)
    asks = json.dumps([[1, 1]] * 10)
    df = pd.DataFrame({'symbol':['BTCUSDT'], 'timestamp':[ts], 'bids':[bids], 'asks':[asks]})
    df.to_sql('order_book', engine, index=False, if_exists='append')

    imbal = dl.get_latest_order_book_imbalance('BTCUSDT')
    assert imbal == pytest.approx((20 - 10) / 30)

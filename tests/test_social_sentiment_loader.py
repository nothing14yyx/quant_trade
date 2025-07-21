import pandas as pd
import sqlalchemy

from quant_trade.data_loader import DataLoader
from quant_trade.social_sentiment_loader import SocialSentimentLoader


def test_update_social_sentiment(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE social_sentiment (date TEXT PRIMARY KEY, score REAL)"
        )

    dl = DataLoader.__new__(DataLoader)
    dl.engine = engine
    dl.retries = 1
    dl.backoff = 0

    df = pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'score': [0.5]})

    def fake_init(self, engine, api_key='', retries=3, backoff=1.0):
        self.engine = engine
        self.retries = retries
        self.backoff = backoff
        self.api_key = ''

    monkeypatch.setattr(SocialSentimentLoader, '__init__', fake_init)
    monkeypatch.setattr(SocialSentimentLoader, 'fetch_scores', lambda self, since: df)

    dl.update_social_sentiment()

    out = pd.read_sql('social_sentiment', engine, parse_dates=['date'])
    assert len(out) == 1
    assert out.iloc[0]['score'] == 0.5


def test_social_sentiment_features():
    times = pd.date_range('2020-01-01', periods=5, freq='h')
    df = pd.DataFrame({
        'open': 1,
        'high': 1,
        'low': 1,
        'close': range(1, 6),
        'volume': 1,
        'social_sentiment': range(5)
    }, index=times)

    from quant_trade.utils.helper import calc_features_raw
    from quant_trade.feature_engineering import calc_cross_features

    f1h = calc_features_raw(df, '1h')
    agg = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'social_sentiment': 'mean'
    }
    f4h = calc_features_raw(df.resample('4h').agg(agg), '4h')
    f1d = calc_features_raw(df.resample('1d').agg(agg), 'd1')

    merged = calc_cross_features(f1h, f4h, f1d)
    assert 'social_sentiment_1h' in merged
    assert 'social_sentiment_4h' in merged
    expected = pd.Series(range(5)).rolling(4, min_periods=1).mean()
    assert merged['social_sentiment_4h'].reset_index(drop=True).equals(expected)

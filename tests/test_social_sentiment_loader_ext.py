import pandas as pd
import sqlalchemy
import datetime as dt

from quant_trade.social_sentiment_loader import SocialSentimentLoader


def test_fetch_scores(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    loader = SocialSentimentLoader(engine, api_key='')

    pages = [
        {
            'data': [
                {'published_at': '2024-06-01T01:00:00Z', 'sentiment': 'positive'},
                {'published_at': '2024-06-01T02:00:00Z', 'sentiment': 'negative'},
            ],
            'next_url': True,
        },
        {
            'data': [
                {'published_at': '2024-05-31T03:00:00Z', 'sentiment': 'bullish'},
            ],
            'next_url': None,
        },
    ]
    idx = 0

    def fake_fetch(self, page=1):
        nonlocal idx
        res = pages[idx]
        idx += 1
        return res

    monkeypatch.setattr(SocialSentimentLoader, '_fetch_posts', fake_fetch)

    df = loader.fetch_scores(dt.date(2024, 5, 31))
    assert len(df) == 2
    val1 = df.loc[df['date'] == pd.Timestamp('2024-06-01'), 'score'].iloc[0]
    val2 = df.loc[df['date'] == pd.Timestamp('2024-05-31'), 'score'].iloc[0]
    assert val1 == 0
    assert val2 == 1


def test_neutral_score(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    loader = SocialSentimentLoader(engine, api_key='')

    pages = [
        {
            'data': [
                {'published_at': '2024-06-01T01:00:00Z', 'sentiment': 'neutral'},
            ],
            'next_url': None,
        },
    ]
    idx = 0

    def fake_fetch(self, page=1):
        nonlocal idx
        res = pages[idx]
        idx += 1
        return res

    monkeypatch.setattr(SocialSentimentLoader, '_fetch_posts', fake_fetch)

    df = loader.fetch_scores(dt.date(2024, 5, 31))
    assert len(df) == 1
    val = df.loc[df['date'] == pd.Timestamp('2024-06-01'), 'score'].iloc[0]
    assert val == 0


def test_update_scores_writes(monkeypatch):
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE social_sentiment (date TEXT PRIMARY KEY, score REAL)"
        )
    loader = SocialSentimentLoader(engine, api_key='')

    df = pd.DataFrame({'date': [pd.Timestamp('2024-06-01')], 'score': [0.2]})
    monkeypatch.setattr(SocialSentimentLoader, 'fetch_scores', lambda self, since: df)

    loader.update_scores(dt.date(2024, 6, 1))

    out = pd.read_sql('social_sentiment', engine, parse_dates=['date'])
    assert len(out) == 1
    assert out.iloc[0]['score'] == 0.2

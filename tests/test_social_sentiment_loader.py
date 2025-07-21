import pandas as pd
import sqlalchemy
from types import SimpleNamespace
from quant_trade import run_scheduler

from quant_trade.data_loader import DataLoader
from quant_trade.social_sentiment_loader import SocialSentimentLoader
from quant_trade import data_loader


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
    dl.ss_cfg = {}

    df = pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'score': [0.5]})

    def fake_init(self, engine, api_key='', plan='free', public=True, currencies=None, retries=3, backoff=1.0):
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


def test_sync_all_calls_social_sentiment(monkeypatch):
    dl = DataLoader.__new__(DataLoader)
    dl.main_iv = '1h'
    dl.aux_ivs = []
    dl.get_top_symbols = lambda: []
    dl.update_sentiment = lambda: None
    dl.update_cg_global_metrics = lambda: None
    dl.update_cg_market_data = lambda symbols: None
    dl.update_cg_coin_categories = lambda symbols: None
    dl.update_cg_category_stats = lambda: None
    dl.update_cm_metrics = lambda symbols: None
    dl.update_funding_rate = lambda sym: None
    dl.update_open_interest = lambda sym: None
    dl.incremental_update_klines = lambda sym, iv: None
    called = []
    dl.update_social_sentiment = lambda: called.append(True)

    dl.sync_all(max_workers=1)
    assert called == [True]


def test_scheduler_update_daily_data_calls_social_sentiment():
    called = []
    sched = SimpleNamespace(
        dl=SimpleNamespace(
            update_sentiment=lambda: None,
            update_cg_global_metrics=lambda: None,
            update_cg_market_data=lambda s: None,
            update_cg_coin_categories=lambda s: None,
            update_cg_category_stats=lambda: None,
            update_cm_metrics=lambda s: None,
            update_social_sentiment=lambda: called.append(True),
        ),
        sg=SimpleNamespace(update_market_phase=lambda e: None),
        engine=None,
    )

    run_scheduler.Scheduler.update_daily_data(sched, [])
    assert called == [True]


def test_social_sentiment_config_parsing(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
mysql:
  host: localhost
  user: root
  password: ''
  database: test
social_sentiment:
  api_key: KEY
  plan: developer
  public: false
  currencies:
    - btc
    - eth
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        data_loader, "create_engine", lambda *a, **k: sqlalchemy.create_engine("sqlite:///:memory:")
    )
    class DummyClient:
        def __init__(self, api_key=None, api_secret=None):
            self.session = SimpleNamespace(proxies={})

    monkeypatch.setattr(data_loader, "Client", DummyClient)
    dl = DataLoader(config_path=cfg_path)

    with dl.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE social_sentiment (date TEXT PRIMARY KEY, score REAL)"
        )

    captured = {}

    def fake_init(self, engine, api_key='', plan='free', public=True, currencies=None, retries=3, backoff=1.0):
        captured.update(
            {
                "api_key": api_key,
                "plan": plan,
                "public": public,
                "currencies": currencies,
            }
        )
        self.engine = engine
        self.retries = retries
        self.backoff = backoff

    monkeypatch.setattr(SocialSentimentLoader, "__init__", fake_init)
    monkeypatch.setattr(SocialSentimentLoader, "update_scores", lambda self, since: None)

    dl.update_social_sentiment()

    assert captured == {
        "api_key": "KEY",
        "plan": "developer",
        "public": False,
        "currencies": ["btc", "eth"],
    }

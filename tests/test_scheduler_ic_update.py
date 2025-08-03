import datetime as dt
from concurrent.futures import Future
from types import SimpleNamespace

from quant_trade import run_scheduler


class DummyExecutor:
    def submit(self, func, *args, **kwargs):
        func(*args, **kwargs)
        f = Future()
        f.set_result(None)
        return f

def test_dispatch_ic_update(monkeypatch):
    calls = []
    fixed_now = dt.datetime(2021, 1, 1, 1, 0, tzinfo=dt.timezone.utc)

    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(run_scheduler, "datetime", FixedDateTime)

    dummy = SimpleNamespace(
        scheduler=SimpleNamespace(enterabs=lambda *a, **k: None, queue=[]),
        executor=DummyExecutor(),
        symbols=[],
        dl=SimpleNamespace(get_top_symbols=lambda n=None: []),
        safe_call=lambda func, *a, **k: func(*a, **k),
        update_oi_and_order_book=lambda syms: None,
        update_klines=lambda syms, iv: None,
        update_funding_rates=lambda syms: None,
        update_daily_data=lambda syms: None,
        update_features=lambda: None,
        update_ic_scores_from_db=lambda: calls.append("ic"),
        generate_signals=lambda syms: None,
        next_symbols_refresh=fixed_now + dt.timedelta(hours=1),
        next_ic_update=fixed_now,
        ic_update_interval=1,
        _calc_next_ic_update=lambda now: now + dt.timedelta(hours=1),
        schedule_next=lambda: None,
    )

    run_scheduler.Scheduler.dispatch_tasks(dummy)
    assert calls == ["ic"]


def test_update_ic_scores_group_by(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({
        "open_time": [0, 1, 0, 1],
        "close_time": [0, 1, 0, 1],
        "symbol": ["A", "A", "B", "B"],
        "open": [1, 1, 1, 1],
        "close": [1, 1, 1, 1],
    })

    monkeypatch.setattr(run_scheduler, "text", lambda q: q)
    monkeypatch.setattr(run_scheduler.pd, "read_sql", lambda *a, **k: df)

    captured = {}

    def fake_update_ic_scores(d, group_by=None):
        captured["data"] = d
        captured["group_by"] = group_by

    sched = SimpleNamespace(
        engine=None,
        ic_update_limit=100,
        sg=SimpleNamespace(update_ic_scores=fake_update_ic_scores, current_weights={}),
    )

    run_scheduler.Scheduler.update_ic_scores_from_db(sched)

    assert captured["group_by"] == "symbol"
    assert list(captured["data"].open_time) == sorted(df.open_time.tolist())


import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import datetime as dt
from types import SimpleNamespace
from concurrent.futures import Future

import run_scheduler

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


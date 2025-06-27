from types import SimpleNamespace

from quant_trade import run_scheduler


def test_generate_signals_skip_none(monkeypatch):
    inserted = []

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *args, **kwargs):
            inserted.extend(args[1] if len(args) > 1 else kwargs.get('params', []))

    engine = SimpleNamespace(begin=lambda: DummyConn())

    sched = SimpleNamespace(
        engine=engine,
        sg=SimpleNamespace(generate_signal=lambda *a, **k: None),
        scaler_params=None,
    )

    monkeypatch.setattr(run_scheduler, "prepare_all_features", lambda *a, **k: ({}, {}, {}, {"close": 1}, {}, {}))
    monkeypatch.setattr(run_scheduler, "load_latest_open_interest", lambda *a, **k: {})
    monkeypatch.setattr(run_scheduler, "load_order_book_imbalance", lambda *a, **k: 0)
    monkeypatch.setattr(run_scheduler, "load_global_metrics", lambda *a, **k: {})

    run_scheduler.Scheduler.generate_signals(sched, ["AAA"])
    assert inserted == []


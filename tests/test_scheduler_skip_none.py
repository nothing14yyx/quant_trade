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
        sg=SimpleNamespace(
            generate_signal_batch=lambda *a, **k: [None],
            diagnose=lambda: {},
        ),
        scaler_params=None,
    )

    async def dummy_prepare(*args, **kwargs):
        return {}, {}, {}, {"close": 1}, {}, {}

    async def dummy_oi(*args, **kwargs):
        return {}

    async def dummy_imb(*args, **kwargs):
        return 0

    async def dummy_metrics(*args, **kwargs):
        return {}

    monkeypatch.setattr(run_scheduler, "prepare_all_features_async", dummy_prepare)
    monkeypatch.setattr(run_scheduler, "load_latest_open_interest_async", dummy_oi)
    monkeypatch.setattr(run_scheduler, "load_order_book_imbalance_async", dummy_imb)
    monkeypatch.setattr(run_scheduler, "load_global_metrics_async", dummy_metrics)

    run_scheduler.Scheduler.generate_signals(sched, ["AAA"])
    assert inserted == []


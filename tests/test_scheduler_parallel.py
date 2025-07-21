from types import SimpleNamespace

from quant_trade import run_scheduler


def test_generate_signals_parallel(monkeypatch):
    inserted = []

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *args, **kwargs):
            inserted.extend(args[1])

    engine = SimpleNamespace(begin=lambda: DummyConn())

    sched = SimpleNamespace(
        engine=engine,
        sg=SimpleNamespace(
            generate_signal_batch=lambda f1, f4, fd, **k: [
                {"signal": 1, "score": i + 1} for i in range(len(f1))
            ],
            diagnose=lambda: {},
        ),
        scaler_params=None,
    )

    async def dummy_prepare(engine, sym, params):
        close = 1 if sym == "A" else 2
        return {}, {}, {}, {"close": close}, {}, {}

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

    run_scheduler.Scheduler.generate_signals(sched, ["A", "B"])
    scores = {d["symbol"]: d["score"] for d in inserted}
    assert scores == {"A": 1, "B": 2}

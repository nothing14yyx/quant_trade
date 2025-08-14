import types

from quant_trade.signal.engine import SignalEngine
from quant_trade.robust_signal_generator import RobustSignalGenerator


def test_engine_run_batch_dispatch():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    called = {}

    def fake_batch(f1, f4, fd, f15=None, symbols=None, **kwargs):
        called["called"] = True
        return [{"id": row["id"]} for row in f1]

    rsg.generate_signal_batch = fake_batch

    engine = SignalEngine(
        rsg,
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
    )

    ctx = {
        "features_1h": [{"id": 1}, {"id": 2}],
        "features_4h": [{}, {}],
        "features_d1": [{}, {}],
    }
    res = engine.run(ctx)
    assert called.get("called")
    assert [r["id"] for r in res] == [1, 2]


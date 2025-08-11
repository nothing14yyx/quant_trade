import types
from collections import OrderedDict
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal import FactorScorerImpl


def test_generate_signal_batch_order_and_diagnose():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = OrderedDict()
    rsg.factor_scorer = FactorScorerImpl(rsg)
    def stub_generate_signal(f1, f4, fd, *a, **k):
        rsg._diagnostic = {"id": f1["id"]}
        return {"id": f1["id"]}
    rsg.generate_signal = stub_generate_signal
    feats = [{"id": i} for i in range(3)]
    res = rsg.generate_signal_batch(feats, [{}]*3, [{}]*3, symbols=["A","B","C"])
    assert [r["id"] for r in res] == [0,1,2]
    assert rsg.diagnose() == {"id": 2}


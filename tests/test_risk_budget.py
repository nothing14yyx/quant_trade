import pytest
from quant_trade.tests.test_utils import make_dummy_rsg
from tests.test_overbought_oversold import base_inputs, make_cache


class DummyAccount:
    def __init__(self, equity):
        self.equity = equity

    def day_loss_pct(self):
        return 0


def test_risk_budget_caps_position():
    rsg = make_dummy_rsg()
    rsg.account = DummyAccount(1000)
    rsg.cfg['risk_budget_per_trade'] = 0.0005
    rsg.cfg['max_pos_pct'] = 1.0
    rsg.compute_tp_sl = lambda price, atr, direction, **k: (price * 1.05, 95)
    (
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_f15m,
    ) = base_inputs()
    res = rsg.finalize_position(
        0.5,
        risk_info,
        risk_info['logic_score'],
        risk_info['env_score'],
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        {},
        raw_f15m,
        {}, {}, {},
        short_mom=0,
        ob_imb=0,
        confirm_15m=0,
        extreme_reversal=False,
        cache=make_cache(),
        symbol="BTCUSDT",
    )
    assert res['stop_loss'] == 95
    assert res['position_size'] == pytest.approx(0.1, rel=1e-2)

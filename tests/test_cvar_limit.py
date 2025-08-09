import pytest
from quant_trade.tests.test_utils import make_dummy_rsg
from tests.test_overbought_oversold import make_cache, base_inputs


class DummyAccount:
    def __init__(self, pnl):
        self.pnl_history = pnl

    def day_loss_pct(self):
        return 0.0


def _run_finalize(rsg):
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
    cache = make_cache()
    return rsg.finalize_position(
        0.5,
        risk_info,
        risk_info["logic_score"],
        risk_info["env_score"],
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
        {},
        {},
        {},
        short_mom=0,
        ob_imb=0,
        confirm_15m=0,
        extreme_reversal=False,
        cache=cache,
        symbol=None,
    )


def test_finalize_position_skips_on_negative_cvar():
    rsg = make_dummy_rsg()
    rsg.account = DummyAccount([-0.1, -0.05, 0.02, 0.03])
    rsg.cvar_alpha = 0.25
    assert _run_finalize(rsg) is None


def test_finalize_position_allows_when_cvar_positive():
    rsg = make_dummy_rsg()
    rsg.account = DummyAccount([0.1, 0.05, 0.02, 0.03])
    rsg.cvar_alpha = 0.25
    assert _run_finalize(rsg) is not None

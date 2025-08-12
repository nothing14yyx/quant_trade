import pytest
from quant_trade.tests.test_utils import make_dummy_rsg
from tests.test_overbought_oversold import base_inputs, make_cache


def test_finalize_position_applies_multipliers():
    rsg = make_dummy_rsg()
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
    risk_info["pos_mult"] = 0.0
    risk_info["score_mult"] = 0.0
    risk_info["details"] = {"penalties": ["limit"]}

    res = rsg.finalize_position(
        0.6,
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
        cache=make_cache(),
        symbol="BTCUSDT",
    )

    assert res["position_size"] == 0.0
    assert res["signal"] == 0
    assert res["score"] == 0.0
    assert "limit" in res["zero_reason"]

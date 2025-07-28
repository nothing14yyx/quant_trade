import pytest
from quant_trade.tests.test_utils import make_dummy_rsg
from tests.test_overbought_oversold import make_cache, base_inputs


def test_trend_alignment_penalty_reduces_position():
    rsg = make_dummy_rsg()
    rsg.min_trend_align = 4
    rsg.direction_filters_enabled = True
    rsg.compute_tp_sl = lambda *a, **k: (None, None)

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
    ) = base_inputs(direction=1)

    cache = make_cache()
    res_full = rsg.finalize_position(
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

    rsg2 = make_dummy_rsg()
    rsg2.min_trend_align = 4
    rsg2.direction_filters_enabled = True
    rsg2.compute_tp_sl = lambda *a, **k: (None, None)
    (
        risk_info2,
        ai_scores2,
        fs2,
        scores2,
        std_1h2,
        std_4h2,
        std_d12,
        std_15m2,
        raw_f1h2,
        raw_f4h2,
        raw_f15m2,
    ) = base_inputs(direction=1)
    fs2["4h"]["trend"] = 0
    fs2["d1"]["trend"] = 0

    cache2 = make_cache()
    res_partial = rsg2.finalize_position(
        0.5,
        risk_info2,
        risk_info2["logic_score"],
        risk_info2["env_score"],
        ai_scores2,
        fs2,
        scores2,
        std_1h2,
        std_4h2,
        std_d12,
        std_15m2,
        raw_f1h2,
        raw_f4h2,
        {},
        raw_f15m2,
        {},
        {},
        {},
        short_mom=0,
        ob_imb=0,
        confirm_15m=0,
        extreme_reversal=False,
        cache=cache2,
        symbol=None,
    )

    assert res_full["signal"] == 1
    assert res_partial["signal"] == 1
    assert res_partial["position_size"] < res_full["position_size"]

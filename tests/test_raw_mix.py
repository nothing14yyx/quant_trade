import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.constants import ZeroReason
from tests.test_overbought_oversold import make_cache, base_inputs


def test_raw_and_std_mixed_volume():
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
    ) = base_inputs(direction=1)

    std_1h["vol_ma_ratio_1h"] = 0.05
    std_1h["vol_ratio_1h_4h"] = 0.05
    std_4h["vol_ratio_1h_4h"] = 0.05

    raw_f1h["vol_ma_ratio_1h"] = 1.0
    raw_f1h["vol_ratio_1h_4h"] = 1.0
    raw_f4h["vol_ratio_1h_4h"] = 1.0

    cache = make_cache()
    res = rsg.finalize_position(
        0.11,
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
        ob_imb=0.2,
        confirm_15m=0,
        extreme_reversal=False,
        cache=cache,
        symbol=None,
    )
    assert res["zero_reason"] != ZeroReason.VOL_RATIO.value
    assert res["details"]["vote"]["ob_th"] == pytest.approx(rsg.ob_th_params["min_ob_th"])

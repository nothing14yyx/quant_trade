import json
from tests.test_overbought_oversold import make_cache, base_inputs
from quant_trade.tests.test_utils import make_dummy_rsg


def test_factor_breakdown_serializable():
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
    res = rsg.finalize_position(
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
        cache=make_cache(),
        symbol="BTCUSDT",
    )
    fb = res.get("factor_breakdown")
    assert fb is not None
    assert set(fb) == {"ai", "trend", "momentum", "volatility", "volume", "sentiment", "funding"}
    json.dumps(res)

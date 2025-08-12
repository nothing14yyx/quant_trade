from quant_trade.tests.test_utils import make_dummy_rsg
from tests.test_overbought_oversold import base_inputs, make_cache


def setup_rsg(min_vote=3, conf_vote=0.2, weight_ai=2):
    rsg = make_dummy_rsg()
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.0
    rsg.vote_weights = {"ai": weight_ai}
    rsg.signal_filters = {"min_vote": min_vote, "confidence_vote": conf_vote, "conf_min": 0.0}
    rsg.vote_params["strong_min"] = 5
    return rsg


def prepare_inputs(direction=1, breakout=0):
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
    ) = base_inputs(direction=direction)
    fs["1h"]["trend"] = 0
    fs["4h"]["trend"] = 0
    fs["d1"]["trend"] = 0
    std_1h["vol_breakout_1h"] = breakout
    raw_f1h["vol_breakout_1h"] = breakout
    return (
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
    )


def test_vote_filter_blocks_signal():
    rsg = setup_rsg()
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
    ) = prepare_inputs(breakout=0)
    cache = make_cache()
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
        cache=cache,
        symbol=None,
    )
    assert res["signal"] == 0
    assert res["position_size"] == 0.0


def test_vote_filter_allows_signal():
    rsg = setup_rsg(weight_ai=6)
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
    ) = prepare_inputs(breakout=1)
    cache = make_cache()
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
        cache=cache,
        symbol=None,
    )
    assert res["signal"] == 1
    assert res["position_size"] > 0

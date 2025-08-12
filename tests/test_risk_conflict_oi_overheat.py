import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.robust_signal_generator import SignalThresholdParams
from quant_trade.constants import RiskReason
from tests.test_overbought_oversold import base_inputs, make_cache


def test_funding_conflict_and_oi_overheat_finalization():
    rsg = make_dummy_rsg()
    rsg.filter_penalty_mode = True
    rsg.veto_conflict_count = 1
    rsg.dynamic_threshold_enabled = False
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.0
    rsg.signal_params = SignalThresholdParams.from_cfg(rsg.signal_threshold_cfg)
    rsg.oi_scale = 0.8
    rsg.penalty_factor = 0.5

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

    fused_score, oi_overheat = rsg.risk_filters.apply_oi_overheat_protection(0.7, 0.4, 0.3)
    cache = make_cache()
    cache.update({"oi_overheat": oi_overheat, "th_oi": 0.3, "oi_chg": 0.4})

    raw_f1h_conflict = dict(raw_f1h)
    raw_f1h_conflict["funding_rate_1h"] = -0.0006

    ret = rsg.risk_filters.apply_risk_filters(
        fused_score=fused_score,
        logic_score=risk_info["logic_score"],
        env_score=risk_info["env_score"],
        std_1h=std_1h,
        std_4h=std_4h,
        std_d1=std_d1,
        raw_f1h=raw_f1h_conflict,
        raw_f4h=raw_f4h,
        raw_fd1={},
        vol_preds={},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache=cache,
        global_metrics=None,
        symbol="BTCUSDT",
    )
    assert ret is not None
    score_mult, pos_mult, reasons = ret
    assert RiskReason.OI_OVERHEAT.value in reasons
    assert RiskReason.FUNDING_PENALTY.value in reasons

    risk_info_base = {
        "fused_score": fused_score * score_mult,
        "risk_score": 0.0,
        "crowding_factor": 1.0,
        "crowding_adjusted": True,
        "risk_th": 0.0,
        "rev_boost": 0.0,
        "oi_overheat": True,
        "th_oi": 0.3,
        "oi_chg": 0.4,
        "funding_conflicts": 1,
        "details": {"penalties": reasons},
        "logic_score": risk_info["logic_score"],
        "env_score": risk_info["env_score"],
    }

    base_res = rsg.finalize_position(
        risk_info_base["fused_score"],
        {**risk_info_base, "score_mult": 1.0, "pos_mult": 1.0},
        risk_info["logic_score"],
        risk_info["env_score"],
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h_conflict,
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

    final_res = rsg.finalize_position(
        risk_info_base["fused_score"],
        {**risk_info_base, "score_mult": score_mult, "pos_mult": pos_mult},
        risk_info["logic_score"],
        risk_info["env_score"],
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h_conflict,
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

    assert final_res["score"] == pytest.approx(base_res["score"] * score_mult, rel=1e-6)
    assert final_res["position_size"] == pytest.approx(
        base_res["position_size"] * pos_mult, rel=1e-6
    )
    assert {
        RiskReason.OI_OVERHEAT.value,
        RiskReason.FUNDING_PENALTY.value,
    }.issubset(set(final_res["details"].get("penalties", [])))

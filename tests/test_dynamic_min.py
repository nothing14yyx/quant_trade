import pytest
from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.constants import ZeroReason


def test_dynamic_min_risk_scaling():
    rsg = make_dummy_rsg()
    cfg = {"min_pos": 0.05}
    params = dict(
        grad_dir=1.0,
        base_coeff=0.2,
        confidence_factor=1.0,
        vol_ratio=1.0,
        fused_score=0.2,
        base_th=0.1,
        regime="trend",
        vol_p=None,
        atr=0.0,
        risk_score=0.0,
        crowding_factor=1.0,
        cfg_th_sig=cfg,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )
    pos_normal, direction_normal, _, zero_reason_normal = rsg.compute_position_size(**params)
    assert pos_normal > 0
    assert zero_reason_normal is None

    params["risk_score"] = 2.0
    pos_high, direction_high, _, zero_reason_high = rsg.compute_position_size(**params)
    assert pos_high > 0
    assert direction_high == 1
    assert zero_reason_high == ZeroReason.MIN_POS.value

    rsg.risk_filters_enabled = False
    pos_off, direction_off, _, zero_reason_off = rsg.compute_position_size(**params)
    assert pos_off > 0
    assert direction_off != 0
    assert zero_reason_off is None


def test_min_pos_vol_scaling():
    rsg = make_dummy_rsg()
    rsg.min_pos_vol_scale = 5.0
    cfg = {"min_pos": 0.05}
    params = dict(
        grad_dir=1.0,
        base_coeff=0.2,
        confidence_factor=1.0,
        vol_ratio=1.0,
        fused_score=0.2,
        base_th=0.1,
        regime="trend",
        vol_p=None,
        atr=0.5,
        risk_score=0.0,
        crowding_factor=1.0,
        cfg_th_sig=cfg,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )
    pos_high, _, _, reason_high = rsg.compute_position_size(**params)
    assert pos_high > 0
    assert reason_high == ZeroReason.MIN_POS.value

    rsg.min_pos_vol_scale = 0.0
    params["atr"] = 0.0
    pos_normal, _, _, reason_normal = rsg.compute_position_size(**params)
    assert pos_normal > 0
    assert pos_high >= pos_normal
    assert reason_normal is None


def test_dyn_th_active_when_filters_off():
    rsg = make_dummy_rsg()
    rsg.risk_filters_enabled = False
    rsg.dynamic_threshold_enabled = True
    rsg.dynamic_threshold = lambda *a, **k: (0.2, 0.0)
    rsg.detect_market_regime = lambda *a, **k: "range"
    cache = {"history_scores": rsg.history_scores, "_raw_history": {"1h": []}, "oi_change_history": []}
    res = rsg.risk_filters.apply_risk_filters(
        fused_score=0.1,
        logic_score=0.1,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h={},
        raw_f4h={},
        raw_fd1={},
        vol_preds={},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache=cache,
        global_metrics=None,
        symbol=None,
    )
    assert res["base_th"] == 0.2

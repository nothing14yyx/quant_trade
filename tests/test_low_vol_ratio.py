import pytest
import math
from quant_trade.tests.test_utils import make_dummy_rsg


def test_low_vol_ratio_config():
    rsg = make_dummy_rsg()
    cfg_th = {'min_pos': 0.0}
    base_params = dict(
        grad_dir=1.0,
        base_coeff=0.4,
        confidence_factor=1.0,
        fused_score=0.11,
        base_th=0.1,
        regime='range',
        vol_p=None,
        atr=0.0,
        risk_score=0.5,
        crowding_factor=1.0,
        cfg_th_sig=cfg_th,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )

    rsg.low_vol_ratio = 0.4
    pos1, _, _, _ = rsg.position_sizer.decide(vol_ratio=0.35, **base_params)

    base_params_no_risk = base_params.copy()
    base_params_no_risk["risk_score"] = 0.0
    pos_no_risk, _, _, _ = rsg.position_sizer.decide(vol_ratio=0.35, **base_params_no_risk)

    rsg.low_vol_ratio = 0.2
    pos2, _, _, _ = rsg.position_sizer.decide(vol_ratio=0.35, **base_params)

    assert pos1 == pytest.approx(pos2 * 0.5)
    assert pos1 == pytest.approx(pos_no_risk * math.exp(-rsg.risk_scale * 0.5))


def test_low_vol_penalty_only_in_range():
    rsg = make_dummy_rsg()
    cfg_th = {"min_pos": 0.0}
    base = dict(
        grad_dir=1.0,
        base_coeff=0.4,
        confidence_factor=1.0,
        fused_score=0.11,
        base_th=0.1,
        vol_p=None,
        atr=0.0,
        risk_score=0.0,
        crowding_factor=1.0,
        cfg_th_sig=cfg_th,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
        vol_ratio=0.35,
    )
    rsg.low_vol_ratio = 0.4
    params_range = base | {"regime": "range"}
    pos_range, _, _, _ = rsg.position_sizer.decide(**params_range)
    params_trend = base | {"regime": "trend"}
    pos_trend, _, _, _ = rsg.position_sizer.decide(**params_trend)
    params_high = base | {"regime": "range", "vol_ratio": 0.6}
    pos_high, _, _, _ = rsg.position_sizer.decide(**params_high)
    assert pos_range == pytest.approx(pos_trend * 0.5)
    assert pos_high == pytest.approx(pos_trend)


def test_vol_prediction_adjustment():
    rsg = make_dummy_rsg()
    cfg_th = {"min_pos": 0.0}
    base = dict(
        grad_dir=1.0,
        base_coeff=0.4,
        confidence_factor=1.0,
        vol_ratio=1.0,
        fused_score=0.3,
        base_th=0.1,
        regime="trend",
        atr=0.0,
        risk_score=0.0,
        crowding_factor=1.0,
        cfg_th_sig=cfg_th,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )
    pos_none, _, _, _ = rsg.position_sizer.decide(vol_p=None, **base)
    pos_half, _, _, _ = rsg.position_sizer.decide(vol_p=0.5, **base)
    pos_floor, _, _, _ = rsg.position_sizer.decide(vol_p=0.9, **base)
    assert pos_half == pytest.approx(pos_none * 0.5)
    assert pos_floor == pytest.approx(pos_none * 0.4)

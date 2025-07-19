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
    assert pos_high == 0.0
    assert direction_high == 0
    assert zero_reason_high == ZeroReason.MIN_POS.value

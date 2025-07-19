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
        risk_score=0.5,
        crowding_factor=1.0,
        cfg_th_sig=cfg_th,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )

    rsg.low_vol_ratio = 0.4
    pos1, _, _, _ = rsg.compute_position_size(vol_ratio=0.35, **base_params)

    base_params_no_risk = base_params.copy()
    base_params_no_risk["risk_score"] = 0.0
    pos_no_risk, _, _, _ = rsg.compute_position_size(vol_ratio=0.35, **base_params_no_risk)

    rsg.low_vol_ratio = 0.2
    pos2, _, _, _ = rsg.compute_position_size(vol_ratio=0.35, **base_params)

    assert pos1 == pytest.approx(pos2 * 0.5)
    assert pos1 == pytest.approx(pos_no_risk * math.exp(-rsg.risk_scale * 0.5))

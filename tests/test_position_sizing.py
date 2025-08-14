import pytest
from types import SimpleNamespace
from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.constants import RiskReason


def test_apply_normalized_multipliers_low_volume():
    rsg = make_dummy_rsg()
    rsg.low_vol_ratio = 0.4
    pos = 1.0
    adj, low_vol = rsg.position_sizer._apply_normalized_multipliers(
        pos,
        regime="range",
        vol_ratio=0.2,
        fused_score=0.1,
        base_th=0.2,
        consensus_all=False,
        vol_p=0.5,
    )
    assert low_vol is True
    assert adj == pytest.approx(pos * 0.25)
    assert adj < pos


def test_decide_raises_to_min_pos():
    rsg = make_dummy_rsg()
    rsg.signal_params = SimpleNamespace(min_pos=0.1)
    rsg.risk_filters_enabled = True
    cfg = {"min_pos": 0.1}
    params = dict(
        grad_dir=1.0,
        base_coeff=0.05,
        confidence_factor=0.0,
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
    pos, direction, _, reason = rsg.position_sizer.decide(**params)
    assert direction == 1
    assert pos == pytest.approx(0.1)
    assert reason == RiskReason.MIN_POS.value

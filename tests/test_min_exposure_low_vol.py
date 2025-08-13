import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.signal.position_sizing import calc_position_size
from quant_trade.constants import RiskReason


def test_min_exposure_applied_under_low_vol():
    """低波动且方向一致时应保持最小敞口"""
    rsg = make_dummy_rsg()
    rsg.low_vol_ratio = 0.4
    base_th = 0.1
    fused_score = 0.09  # 略低于阈值
    min_pos = 0.05
    vol_ratio = 0.2  # 低于 low_vol_ratio 触发低波动惩罚

    old_pos = calc_position_size(
        fused_score,
        base_th,
        max_position=rsg.max_position,
        gamma=0.05,
        min_exposure=min_pos,
    )

    params = dict(
        grad_dir=1.0,
        base_coeff=0.1,
        confidence_factor=1.0,
        vol_ratio=vol_ratio,
        fused_score=fused_score,
        base_th=base_th,
        regime="range",
        vol_p=None,
        atr=0.0,
        risk_score=0.0,
        crowding_factor=1.0,
        cfg_th_sig={"min_pos": min_pos},
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )

    new_pos, direction, _, reason = rsg.position_sizer.decide(**params)

    assert old_pos == 0.0
    assert direction == 1
    assert new_pos >= pytest.approx(min_pos)
    assert reason == RiskReason.MIN_POS.value

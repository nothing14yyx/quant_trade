import pytest
from collections import deque

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.robust_signal_generator import SignalThresholdParams
from quant_trade.constants import ZeroReason


def make_cache():
    return {
        "history_scores": deque(),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4)},
    }


def test_penalty_on_risk_filters():
    rsg = make_dummy_rsg()
    rsg.filter_penalty_mode = True
    rsg.penalty_factor = 0.5
    rsg.veto_conflict_count = 1
    rsg.risk_score_limit = 0.5
    rsg.dynamic_threshold_enabled = False
    rsg.signal_params = SignalThresholdParams.from_cfg(rsg.signal_threshold_cfg)
    rsg.risk_manager.calc_risk = lambda *a, **k: 1.0
    cache = make_cache()
    raw_f1h = {"funding_rate_1h": -0.001}
    res = rsg.risk_filters.apply_risk_filters(
        fused_score=0.5,
        logic_score=0.5,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h=raw_f1h,
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
    assert res is not None
    assert pytest.approx(res["fused_score"], rel=1e-6) == 0.01225
    penalties = res["details"]["penalties"]
    assert ZeroReason.FUNDING_PENALTY.value in penalties
    assert ZeroReason.RISK_PENALTY.value in penalties


def test_position_penalty_mode():
    rsg = make_dummy_rsg()
    rsg.filter_penalty_mode = True
    rsg.penalty_factor = 0.5
    rsg.veto_level = 0.4
    pos, direction, zr, p1 = rsg._apply_position_filters(
        0.2,
        1,
        weak_vote=True,
        funding_conflicts=0,
        oi_overheat=False,
        risk_score=0.0,
        logic_score=0.0,
        base_th=0.1,
        conflict_filter_triggered=False,
        zero_reason=None,
    )
    assert pos == pytest.approx(0.1)
    assert direction == 1
    assert zr is None
    assert p1 == [ZeroReason.VOTE_PENALTY.value]

    pos2, direction2, zr2, p2 = rsg._apply_position_filters(
        0.2,
        1,
        weak_vote=False,
        funding_conflicts=1,
        oi_overheat=False,
        risk_score=0.0,
        logic_score=0.0,
        base_th=0.1,
        conflict_filter_triggered=True,
        zero_reason=None,
    )
    assert pos2 == pytest.approx(0.2 * 0.5 * 0.5)
    assert direction2 == 1
    assert zr2 is None
    assert set(p2) == {ZeroReason.FUNDING_PENALTY.value, ZeroReason.CONFLICT_PENALTY.value}

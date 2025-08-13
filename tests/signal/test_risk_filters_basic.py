import pytest
from collections import deque

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.signal import compute_dynamic_threshold
from quant_trade.constants import RiskReason


def test_oi_overheat_reason():
    rsg = make_dummy_rsg()
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.0
    cache = {
        "history_scores": deque(),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4), "d1": []},
        "oi_overheat": True,
    }
    params = rsg.signal_params
    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    score_mult, pos_mult = rsg.risk_filters.compute_risk_multipliers(
        fused_score=0.5,
        logic_score=0.5,
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
        dyn_base=dyn_base,
    )
    reasons = rsg.risk_filters.collect_risk_reasons(
        0.5, score_mult, pos_mult, cache
    )
    assert (score_mult, pos_mult) == (1.0, 1.0)
    assert reasons == [RiskReason.OI_OVERHEAT.value]


def test_crowding_conflict_reason(monkeypatch):
    rsg = make_dummy_rsg()
    rsg.filter_penalty_mode = True
    rsg.penalty_factor = 0.5
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.0
    cache = {
        "history_scores": deque(),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4), "d1": []},
    }
    params = rsg.signal_params
    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )

    def fake_crowding(
        fused_score,
        *,
        base_th,
        all_scores_list,
        oi_chg,
        cache,
        vol_pred,
        oi_overheat,
        symbol,
    ):
        return fused_score, 2.0, None

    monkeypatch.setattr(
        rsg.risk_filters, "apply_crowding_protection", fake_crowding
    )
    score_mult, pos_mult = rsg.risk_filters.compute_risk_multipliers(
        fused_score=1.0,
        logic_score=1.0,
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
        dyn_base=dyn_base,
    )
    reasons = rsg.risk_filters.collect_risk_reasons(
        1.0, score_mult, pos_mult, cache
    )
    assert score_mult == pytest.approx(1.0)
    assert pos_mult == pytest.approx(0.5)
    assert reasons == [RiskReason.CROWDING_PENALTY.value]


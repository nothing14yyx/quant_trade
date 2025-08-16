import pytest
from collections import deque

from quant_trade.tests.test_utils import make_dummy_rsg


def _make_cache():
    return {
        "history_scores": deque(),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4), "d1": []},
    }


@pytest.mark.parametrize(
    "scope, expected_score, expected_pos",
    [
        ("score", 0.55, 1.0),
        ("pos", 1.0, 0.55),
        ("both", 0.55, 0.55),
    ],
)
def test_risk_score_scope(scope, expected_score, expected_pos):
    rsg = make_dummy_rsg()
    rsg.risk_rule_scopes = {"risk_score": scope}
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.5
    cache = _make_cache()
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
        dyn_base=None,
    )
    assert pytest.approx(score_mult, rel=1e-6) == expected_score
    assert pytest.approx(pos_mult, rel=1e-6) == expected_pos


def test_crowding_scope_pos(monkeypatch):
    rsg = make_dummy_rsg()
    rsg.risk_rule_scopes = {"crowding": "pos"}
    rsg.risk_manager.calc_risk = lambda *a, **k: 0.0
    cache = _make_cache()

    def fake_crowding(fused_score, **kwargs):
        return fused_score, 0.5, None

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
        dyn_base=None,
    )

    assert score_mult == pytest.approx(1.0)
    assert pos_mult == pytest.approx(0.5)


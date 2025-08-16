import threading
from types import SimpleNamespace

import pytest

from quant_trade.signal.risk_filters import RiskFiltersImpl
from quant_trade.constants import RiskReason


class DummyCore(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.risk_filters_enabled = True
        self.dynamic_threshold_enabled = False
        self.signal_params = SimpleNamespace(base_th=0.1, rev_boost=0.0, quantile=0.6)
        self.signal_threshold_cfg = {"base_th": 0.1}
        self.veto_conflict_count = 1
        self.filter_penalty_mode = False
        self.penalty_factor = 0.5
        self.risk_adjust_factor = 0.0
        self.risk_adjust_threshold = 0.0
        self.risk_score_limit = 100.0
        self.risk_th_quantile = 0.6
        self.crowding_limit = 2.0
        self.oi_scale = 0.8
        self.rsi_k = 1.0
        self.crowding_protection = (
            lambda self, all_scores, fused_score, base_th: 0.5
        ).__get__(self, DummyCore)
        self.get_dynamic_oi_threshold = lambda pred_vol=None: 0.1
        self.detect_market_regime = lambda *a, **k: "range"
        self.risk_manager = SimpleNamespace(
            calc_risk=lambda env_score, pred_vol=None, oi_change=None: 0.0
        )
        self._lock = threading.RLock()
        self._cooldown = 0
        self.all_scores_list = []


def make_filters():
    core = DummyCore()
    return RiskFiltersImpl(core)


def base_cache():
    return {
        "history_scores": [],
        "oi_change_history": [],
        "_raw_history": {"1h": [], "d1": []},
    }


def test_funding_rate_conflict():
    rf = make_filters()
    cache = base_cache()
    score_mult, _pos_mult, reasons = rf.apply_risk_filters(
        fused_score=1.0,
        logic_score=1.0,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h={"funding_rate_1h": -0.001},
        raw_f4h={},
        raw_fd1={},
        vol_preds={"1h": 0.0},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache=cache,
        global_metrics=None,
        symbol="BTC",
        dyn_base=None,
    )
    assert score_mult < 1.0
    assert RiskReason.FUNDING_CONFLICT.value in reasons


def test_crowding_protection_reduces_multiplier():
    rf = make_filters()
    cache = {}
    fused, factor, _th = rf.apply_crowding_protection(
        1.0,
        base_th=0.1,
        all_scores_list=[0.2, 0.3, 0.25],
        oi_chg=0.2,
        cache=cache,
        vol_pred=None,
        oi_overheat=False,
        symbol="BTC",
    )
    assert factor < 1.0
    assert fused < 1.0


def test_oi_overheat_reason():
    rf = make_filters()
    fused, overheat = rf.apply_oi_overheat_protection(1.0, 0.5, 0.1)
    assert overheat
    assert fused == pytest.approx(1.0 * rf.core.oi_scale)

    rf.last_funding_conflicts = 0
    rf.last_risk_th = 0.0
    rf.last_risk_score = 0.0
    rf.last_crowding_factor = 1.0

    reasons = rf.collect_risk_reasons(
        fused_score=fused,
        score_mult=1.0,
        pos_mult=1.0,
        cache={"oi_overheat": True},
    )
    assert RiskReason.OI_OVERHEAT.value in reasons


def test_periods_excludes_d1():
    rf = make_filters()
    cache = base_cache()
    score_mult_d1, _pos_mult, _ = rf.apply_risk_filters(
        fused_score=1.0,
        logic_score=1.0,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h={},
        raw_f4h={},
        raw_fd1={"funding_rate_d1": -0.01},
        vol_preds={"1h": 0.0},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache=cache,
        global_metrics=None,
        symbol="BTC",
        dyn_base=None,
    )
    cache2 = base_cache()
    score_mult_no_d1, _pos_mult, _ = rf.apply_risk_filters(
        fused_score=1.0,
        logic_score=1.0,
        env_score=0.0,
        std_1h={},
        std_4h={},
        std_d1={},
        raw_f1h={},
        raw_f4h={},
        raw_fd1={"funding_rate_d1": -0.01},
        vol_preds={"1h": 0.0},
        open_interest=None,
        all_scores_list=None,
        rev_dir=0,
        cache=cache2,
        global_metrics=None,
        symbol="BTC",
        dyn_base=None,
        periods=("1h", "4h"),
    )
    assert score_mult_d1 < 1.0
    assert score_mult_no_d1 == 1.0

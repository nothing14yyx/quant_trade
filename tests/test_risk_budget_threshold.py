import numpy as np
import pytest

from collections import deque

from quant_trade.robust_signal_generator import (
    risk_budget_threshold,
    SignalThresholdParams,
)
from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.signal import compute_dynamic_threshold


def test_risk_budget_threshold_basic():
    data = [0.1, 0.2, 0.15, 0.4, 0.3]
    th = risk_budget_threshold(data, quantile=0.5)
    assert th == pytest.approx(np.quantile(np.abs(data), 0.5))


def test_risk_budget_threshold_in_filter():
    rsg = make_dummy_rsg()
    rsg.risk_adjust_threshold = 0.02
    rsg.signal_params = SignalThresholdParams.from_cfg(rsg.signal_threshold_cfg)
    rsg.oi_change_history.extend([0.05] * 20)
    cache = {
        "history_scores": deque(),
        "oi_change_history": rsg.oi_change_history,
        "_raw_history": {"1h": deque(maxlen=4)},
    }
    params = rsg.signal_params
    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    res = rsg.risk_filters.apply_risk_filters(
        fused_score=0.04,
        logic_score=0.04,
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
    assert res is not None
    score_mult, pos_mult, reasons = res
    assert score_mult == 0.0
    assert pos_mult == 0.0

    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    res2 = rsg.risk_filters.apply_risk_filters(
        fused_score=0.06,
        logic_score=0.06,
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
    assert res2 is not None
    score_mult2, pos_mult2, reasons2 = res2
    assert score_mult2 > 0
    assert pos_mult2 > 0
    assert reasons2 == []


def test_history_quantile_threshold():
    rsg = make_dummy_rsg()
    rsg.risk_adjust_threshold = None
    rsg.risk_th_quantile = 0.5
    rsg.signal_params = SignalThresholdParams.from_cfg(rsg.signal_threshold_cfg)
    cache = {
        "history_scores": deque([0.01, 0.02, -0.05, 0.04]),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4)},
    }

    params = rsg.signal_params
    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    res = rsg.risk_filters.apply_risk_filters(
        fused_score=0.02,
        logic_score=0.02,
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

    assert res is not None
    score_mult, pos_mult, _ = res
    assert score_mult == 0.0
    assert pos_mult == 0.0

    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    res2 = rsg.risk_filters.apply_risk_filters(
        fused_score=0.06,
        logic_score=0.06,
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

    assert res2 is not None
    score_mult2, pos_mult2, reasons2 = res2
    assert score_mult2 > 0
    assert pos_mult2 > 0


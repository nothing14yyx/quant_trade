import logging
from collections import deque

import pytest

from quant_trade.tests.test_utils import make_dummy_rsg
from quant_trade.robust_signal_generator import SignalThresholdParams
from quant_trade.signal import compute_dynamic_threshold


def test_nan_dynamic_risk_threshold(caplog):
    rsg = make_dummy_rsg()
    rsg.dynamic_threshold_enabled = False
    rsg.risk_adjust_threshold = 0.03
    rsg.signal_params = SignalThresholdParams.from_cfg(rsg.signal_threshold_cfg)
    cache = {
        "history_scores": deque(),
        "oi_change_history": deque(),
        "_raw_history": {"1h": deque(maxlen=4)},
    }
    params = rsg.signal_params
    dyn_base = compute_dynamic_threshold(
        cache["history_scores"], params.window, params.dynamic_quantile
    )
    with caplog.at_level(logging.WARNING):
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
    score_mult, pos_mult, _ = res
    assert score_mult > 0
    assert pos_mult > 0
    assert "历史数据不足" in caplog.text

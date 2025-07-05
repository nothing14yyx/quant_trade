import pytest
from collections import deque

from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.tests.test_utils import make_dummy_rsg


def make_cache():
    return {
        "history_scores": deque(maxlen=500),
        "_prev_raw": {"15m": None, "1h": None, "4h": None, "d1": None},
        "_raw_history": {
            "15m": deque(maxlen=4),
            "1h": deque(maxlen=4),
            "4h": deque(maxlen=2),
            "d1": deque(maxlen=2),
        },
    }


def base_inputs(direction=1):
    sign = 1 if direction > 0 else -1
    risk_info = {
        "base_th": 0.1,
        "crowding_factor": 1.0,
        "risk_score": 0.5,
        "regime": "range",
        "rev_dir": 0,
        "funding_conflicts": 0,
        "logic_score": 0.6 * sign,
        "env_score": 1.0,
    }
    ai_scores = {"1h": 0.6 * sign, "4h": 0.6 * sign, "d1": 0.0}
    fs = {
        "1h": {"trend": sign, "momentum": 0, "volatility": 0, "volume": 0, "sentiment": 0, "funding": 0},
        "4h": {"trend": sign, "momentum": 0, "volatility": 0, "volume": 0, "sentiment": 0, "funding": 0},
        "d1": {"trend": sign, "momentum": 0, "volatility": 0, "volume": 0, "sentiment": 0, "funding": 0},
    }
    scores = {"1h": 0.6 * sign, "4h": 0.6 * sign, "d1": 0.6 * sign}
    std_base = {
        "vol_ma_ratio_1h": 1,
        "vol_ratio_1h_4h": 1,
        "supertrend_dir_1h": 0,
        "donchian_perc_1h": 0,
        "atr_pct_1h": 0.01,
        "bb_width_1h": 0.02,
        "vol_breakout_1h": 1,
    }
    std_1h = std_base.copy()
    std_4h = {"supertrend_dir_4h": 0, "vol_ratio_1h_4h": 1, "atr_pct_4h": 0}
    std_d1 = {"supertrend_dir_d1": 0}
    std_15m = {}
    raw_f1h = {
        "close": 100,
        "atr_pct_1h": 0.01,
        "bb_width_1h": 0.02,
        "vol_breakout_1h": 1,
    }
    raw_f4h = {"atr_pct_4h": 0}
    raw_f15m = {}
    return (
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_f15m,
    )


def test_oversold_sets_zero_position():
    rsg = make_dummy_rsg()
    (
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_f15m,
    ) = base_inputs(direction=-1)
    raw_fd1 = {"rsi_d1": 20, "cci_d1": -120}
    cache = make_cache()
    res = rsg.finalize_position(
        -0.6,
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_fd1,
        raw_f15m,
        {},
        {},
        {},
        short_mom=0,
        ob_imb=0,
        confirm_15m=0,
        oversold_reversal=False,
        cache=cache,
        symbol=None,
    )
    assert res["signal"] == -1
    assert 0 < res["position_size"] < 0.15
    assert res["zero_reason"] is None


def test_overbought_sets_zero_position():
    rsg = make_dummy_rsg()
    (
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_f15m,
    ) = base_inputs(direction=1)
    raw_fd1 = {"rsi_d1": 80, "cci_d1": 120}
    cache = make_cache()
    res = rsg.finalize_position(
        0.6,
        risk_info,
        ai_scores,
        fs,
        scores,
        std_1h,
        std_4h,
        std_d1,
        std_15m,
        raw_f1h,
        raw_f4h,
        raw_fd1,
        raw_f15m,
        {},
        {},
        {},
        short_mom=0,
        ob_imb=0,
        confirm_15m=0,
        oversold_reversal=False,
        cache=cache,
        symbol=None,
    )
    assert res["signal"] == 1
    assert 0 < res["position_size"] < 0.15
    assert res["zero_reason"] is None


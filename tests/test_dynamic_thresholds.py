import pytest

from quant_trade.signal import (
    DynamicThresholdInput,
    SignalThresholdParams,
    DynamicThresholdParams,
    calc_dynamic_threshold,
)


def test_calc_dynamic_threshold_volatility_extremes():
    sig_p = SignalThresholdParams(base_th=0.1, low_base=0.0)
    dyn_p = DynamicThresholdParams()

    low = DynamicThresholdInput(
        atr=0.0,
        adx=0.0,
        funding=0.0,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    low_th, low_rb = calc_dynamic_threshold(low)
    assert low_th == pytest.approx(sig_p.base_th)
    assert low_rb == pytest.approx(sig_p.rev_boost)

    high = DynamicThresholdInput(
        atr=0.1,
        adx=50.0,
        funding=0.05,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    high_th, high_rb = calc_dynamic_threshold(high)
    expected_high = sig_p.base_th
    expected_high += min(dyn_p.atr_cap, 0.1 * dyn_p.atr_mult)
    expected_high += min(dyn_p.funding_cap, 0.05 * dyn_p.funding_mult)
    expected_high += min(dyn_p.adx_cap, 50.0 / dyn_p.adx_div)
    assert high_th == pytest.approx(expected_high)
    assert high_rb == pytest.approx(sig_p.rev_boost)
    assert high_th > low_th


def test_regime_and_reversal_adjustments():
    sig_p = SignalThresholdParams(base_th=0.1, rev_boost=0.02, rev_th_mult=1.5)
    dyn_p = DynamicThresholdParams()

    trend_input = DynamicThresholdInput(
        atr=0.0,
        adx=0.0,
        funding=0.0,
        regime="trend",
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    th_trend, rb_trend = calc_dynamic_threshold(trend_input)
    assert th_trend == pytest.approx(sig_p.base_th * 1.05)
    assert rb_trend == pytest.approx(sig_p.rev_boost * 0.8)

    range_input = DynamicThresholdInput(
        atr=0.0,
        adx=0.0,
        funding=0.0,
        regime="range",
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    th_range, rb_range = calc_dynamic_threshold(range_input)
    assert th_range == pytest.approx(sig_p.base_th * 0.95)
    assert rb_range == pytest.approx(sig_p.rev_boost * 1.2)

    rev_input = DynamicThresholdInput(
        atr=0.0,
        adx=0.0,
        funding=0.0,
        regime="trend",
        reversal=True,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    th_rev, rb_rev = calc_dynamic_threshold(rev_input)
    expected_th_rev = sig_p.base_th * sig_p.rev_th_mult * 1.05
    expected_rb_rev = sig_p.rev_boost * 0.8
    assert th_rev == pytest.approx(expected_th_rev)
    assert rb_rev == pytest.approx(expected_rb_rev)

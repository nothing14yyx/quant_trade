import pytest

from quant_trade.signal import (
    DynamicThresholdInput,
    SignalThresholdParams,
    DynamicThresholdParams,
    calc_dynamic_threshold,
)


def test_history_percentile_base():
    params = SignalThresholdParams(base_th=0.05)
    history = [0.1] * 100
    data = DynamicThresholdInput(
        atr=0,
        adx=0,
        history_scores=history,
        base=0.05,
        signal_params=params,
    )
    th, rb = calc_dynamic_threshold(data)
    assert th == pytest.approx(0.1)
    assert rb == pytest.approx(params.rev_boost)


def test_atr_funding_stack():
    sig_p = SignalThresholdParams(base_th=0, low_base=0)
    dyn_p = DynamicThresholdParams()

    atr_only = DynamicThresholdInput(
        atr=0.02,
        adx=0,
        funding=0,
        atr_4h=0.01,
        atr_d1=0.01,
        base=0,
        low_base=0,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    fund_only = DynamicThresholdInput(
        atr=0,
        adx=0,
        funding=0.01,
        pred_vol=0.02,
        pred_vol_4h=0.01,
        pred_vol_d1=0.01,
        vix_proxy=0.02,
        base=0,
        low_base=0,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    both = DynamicThresholdInput(
        atr=0.02,
        adx=0,
        funding=0.01,
        atr_4h=0.01,
        atr_d1=0.01,
        pred_vol=0.02,
        pred_vol_4h=0.01,
        pred_vol_d1=0.01,
        vix_proxy=0.02,
        base=0,
        low_base=0,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    th_atr, _ = calc_dynamic_threshold(atr_only)
    th_fund, _ = calc_dynamic_threshold(fund_only)
    th_both, _ = calc_dynamic_threshold(both)
    assert th_atr == pytest.approx(0.10)
    assert th_fund == pytest.approx(0.08)
    assert th_both == pytest.approx(th_atr + th_fund)

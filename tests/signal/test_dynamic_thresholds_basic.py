import pytest

from quant_trade.signal import (
    DynamicThresholdInput,
    SignalThresholdParams,
    DynamicThresholdParams,
    calc_dynamic_threshold,
    compute_dynamic_threshold,
    ThresholdParams,
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


def test_extra_proxies_increase_threshold():
    sig_p = SignalThresholdParams(base_th=0, low_base=0)
    dyn_p = DynamicThresholdParams()
    extra = DynamicThresholdInput(
        atr=0,
        adx=0,
        funding=0,
        iv_proxy=0.02,
        macro_proxy=0.02,
        onchain_proxy=0.02,
        base=0,
        low_base=0,
        signal_params=sig_p,
        dynamic_params=dyn_p,
    )
    th, _ = calc_dynamic_threshold(extra)
    expected = min(dyn_p.funding_cap, (0.25 * 0.02 * 3) * dyn_p.funding_mult)
    assert th == pytest.approx(expected)


def test_phase_mult_with_history_scores():
    params = ThresholdParams(base_th=0.0, low_base=0.0, rev_boost=0.02)
    data = {
        "atr": 0.01,
        "atr_4h": 0.01,
        "atr_d1": 0.01,
        "adx": 30,
        "adx_4h": 20,
        "adx_d1": 10,
        "funding": 0.01,
        "pred_vol": 0.02,
        "pred_vol_4h": 0.01,
        "pred_vol_d1": 0.01,
        "vix_proxy": 0.02,
        "phase": "trend",
    }
    hist = [0.1] * 80
    phase_mult = {
        "4h": 0.4,
        "d1": 0.2,
        "pred_vol": 0.6,
        "pred_vol_4h": 0.3,
        "pred_vol_d1": 0.2,
        "vix_proxy": 0.5,
    }
    th, rb = compute_dynamic_threshold(data, params, hist, phase_mult)

    atr_eff = 0.01 + 0.4 * 0.01 + 0.2 * 0.01
    atr_part = min(params.atr_cap, atr_eff * params.atr_mult)
    fund_eff = (
        0.01
        + 0.6 * 0.02
        + 0.3 * 0.01
        + 0.2 * 0.01
        + 0.5 * 0.02
    )
    fund_part = min(params.funding_cap, fund_eff * params.funding_mult)
    adx_eff = 30 + 0.4 * 20 + 0.2 * 10
    adx_part = min(params.adx_cap, adx_eff / params.adx_div)
    base_before = 0.1 + atr_part + fund_part + adx_part
    expected_th = base_before * 1.05
    expected_rb = params.rev_boost * 0.8
    assert th == pytest.approx(expected_th, rel=1e-4)
    assert rb == pytest.approx(expected_rb)


@pytest.mark.parametrize(
    "phase,reversal,th_mult,rb_mult",
    [
        ("trend", False, 1.05, 0.8),
        ("range", False, 0.95, 1.2),
        ("unknown", True, 1.1, 1.0),
    ],
)
def test_regime_and_reversal(phase, reversal, th_mult, rb_mult):
    params = ThresholdParams(
        base_th=0.1, low_base=0.0, rev_boost=0.02, rev_th_mult=1.1
    )
    data = {"atr": 0.0, "adx": 0.0, "funding": 0.0, "phase": phase, "reversal": reversal}
    th, rb = compute_dynamic_threshold(data, params)
    assert th == pytest.approx(params.base_th * th_mult)
    assert rb == pytest.approx(params.rev_boost * rb_mult)

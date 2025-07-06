import pandas as pd
import pandas_ta as ta
import pytest
import importlib.util
from pathlib import Path
import numpy as np

helper_path = Path(__file__).resolve().parents[1] / "quant_trade" / "utils" / "helper.py"
spec = importlib.util.spec_from_file_location("helper", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

_safe_ta = helper._safe_ta
calc_mfi_np = helper.calc_mfi_np
calc_price_channel = helper.calc_price_channel
calc_support_resistance = helper.calc_support_resistance


def test_vwap_np_window():
    high = [1, 2, 3, 4]
    low = [0, 1, 2, 3]
    close = [0.5, 1.5, 2.5, 3.5]
    volume = [10, 10, 10, 10]
    res_all = helper.vwap_np(high, low, close, volume)
    res_win2 = helper.vwap_np(high, low, close, volume, window=2)
    assert res_all.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert res_win2.tolist() == pytest.approx([0.5, 1.0, 2.0, 3.0])


def test_safe_ta_with_short_series():
    s = pd.Series([1, 2], index=pd.date_range('2020-01-01', periods=2, freq='h'))
    df = _safe_ta(ta.macd, s, index=s.index)
    assert isinstance(df, pd.DataFrame)
    assert (df == 0).all().all()


def test_calc_mfi_np_empty():
    ratio, mfi = calc_mfi_np([], [], [], [])
    assert len(ratio) == 0
    assert len(mfi) == 0


def test_calc_mfi_np_all_nan_no_warning():
    arr = np.array([float('nan')] * 5)
    with pytest.warns(None):
        ratio, mfi = calc_mfi_np(arr, arr, arr, arr)
    assert len(ratio) == 5
    assert len(mfi) == 5


def test_calc_price_channel_with_nan():
    idx = pd.date_range('2020-01-01', periods=3, freq='h')
    high = pd.Series([1.0, float('nan'), 3.0], index=idx)
    low = pd.Series([0.5, float('nan'), 2.0], index=idx)
    close = pd.Series([0.8, 1.0, 2.5], index=idx)
    ch = calc_price_channel(high, low, close, window=2)
    assert ch['channel_pos'].iloc[0] == pytest.approx(0.6)
    assert ch['upper'].iloc[1] == pytest.approx(1.0)


def test_calc_price_channel_all_nan():
    idx = pd.date_range('2020-01-01', periods=3, freq='h')
    s = pd.Series([float('nan')] * 3, index=idx)
    with pytest.warns(None):
        ch = calc_price_channel(s, s, s, window=2)
    assert ch.isna().all().all()


def test_calc_support_resistance_basic():
    idx = pd.date_range('2020-01-01', periods=4, freq='h')
    high = pd.Series([1, 1, 1, 1], index=idx)
    low = pd.Series([1, 1, 1, 1], index=idx)
    close = pd.Series([1, 0.9, 1.1, 0.8], index=idx)
    sr = calc_support_resistance(high, low, close, window=2)
    assert sr['break_support'].iloc[1] == 1.0
    assert sr['break_resistance'].iloc[2] == 1.0


def test_calc_features_raw_support_resistance():
    times = pd.date_range('2020-01-01', periods=3, freq='h')
    df = pd.DataFrame({
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [100, 100, 100],
    }, index=times)
    feats = helper.calc_features_raw(df, '1h')
    for col in (
        'support_level_1h',
        'resistance_level_1h',
        'break_support_1h',
        'break_resistance_1h',
    ):
        assert col in feats

def test_calc_features_raw_vwap_window():
    times = pd.date_range('2020-01-01', periods=4, freq='h')
    df = pd.DataFrame({
        'open': [1, 2, 3, 4],
        'high': [1, 2, 3, 4],
        'low': [0, 1, 2, 3],
        'close': [0.5, 1.5, 2.5, 3.5],
        'volume': [10, 10, 10, 10],
    }, index=times)

    feats_all = helper.calc_features_raw(df, '1h')
    feats_win2 = helper.calc_features_raw(df, '1h', vwap_window=2)
    expected_all = helper.vwap_np(
        feats_all['high'],
        feats_all['low'],
        feats_all['close'],
        feats_all['volume'],
    )
    expected_win2 = helper.vwap_np(
        feats_win2['high'],
        feats_win2['low'],
        feats_win2['close'],
        feats_win2['volume'],
        window=2,
    )
    assert feats_all['vwap_1h'].iloc[-1] == pytest.approx(expected_all[-1])
    assert feats_win2['vwap_1h'].iloc[-1] == pytest.approx(expected_win2[-1])


def test_calc_rsi_divergence():
    times = pd.date_range('2020-01-01', periods=15, freq='h')
    close = pd.Series(range(1, 16), index=times)
    rsi = pd.Series(
        [40, 45, 50, 55, 60, 65, 70, 75, 80, 78, 77, 76, 75, 74, 70],
        index=times,
    )
    div = helper.calc_rsi_divergence(close, rsi)
    assert div['bear'].iloc[-1] == 1.0
    assert div['bull'].sum() == 0

    times2 = pd.date_range('2020-01-01', periods=16, freq='h')
    close2 = pd.Series(
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5], index=times2
    )
    rsi2 = pd.Series(
        [60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 20, 30],
        index=times2,
    )
    div2 = helper.calc_rsi_divergence(close2, rsi2)
    assert div2['bull'].iloc[-1] == 1.0
    assert div2['bear'].sum() == 0


def test_calc_features_raw_rsi_divergence():
    times = pd.date_range('2020-01-01', periods=15, freq='h')
    close = pd.Series(range(1, 16), index=times)
    df = pd.DataFrame(
        {
            'open': close,
            'high': close,
            'low': close,
            'close': close,
            'volume': [1] * len(close),
        },
        index=times,
    )
    feats = helper.calc_features_raw(df, '1h')
    assert 'rsi_bull_div_1h' in feats
    assert 'rsi_bear_div_1h' in feats




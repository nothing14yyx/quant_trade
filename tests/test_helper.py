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


def test_safe_ta_with_short_series():
    s = pd.Series([1, 2], index=pd.date_range('2020-01-01', periods=2, freq='h'))
    df = _safe_ta(ta.macd, s, index=s.index)
    assert isinstance(df, pd.DataFrame)
    assert df.isna().all().all()


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



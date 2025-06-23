import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pandas_ta as ta

from utils.helper import _safe_ta, calc_mfi_np


def test_safe_ta_with_short_series():
    s = pd.Series([1, 2], index=pd.date_range('2020-01-01', periods=2, freq='h'))
    df = _safe_ta(ta.macd, s, index=s.index)
    assert isinstance(df, pd.DataFrame)
    assert df.isna().all().all()


def test_calc_mfi_np_empty():
    ratio, mfi = calc_mfi_np([], [], [], [])
    assert len(ratio) == 0
    assert len(mfi) == 0

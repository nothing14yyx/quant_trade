import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from utils.feature_health import apply_health_check_df

def test_apply_health_check_df_datetime():
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=2, freq='h'),
        'value': [1e5, -1e5],
        'other': [float('inf'), float('-inf')]
    })
    res = apply_health_check_df(df)
    # time 列应保持不变
    assert res['time'].equals(df['time'])
    # 数值列应被 clip 至 [-1e4, 1e4]
    assert res['value'].tolist() == [1e4, -1e4]
    assert res['other'].tolist() == [1e4, -1e4]
    # 确保生成 _isnan 标志列
    for c in ['time_isnan', 'value_isnan', 'other_isnan']:
        assert c in res

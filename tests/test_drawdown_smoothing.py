import pandas as pd
import numpy as np
from quant_trade.feature_engineering import FeatureEngineer


def test_d1_drawdown_smoothing():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.rise_transform = 'none'
    fe.boxcox_lambda = {}
    df = pd.DataFrame({
        'open_time': pd.date_range('2020-01-01', periods=3, freq='D'),
        'close': [3.0, 2.0, 1.0],
        'symbol': ['BTC'] * 3,
    })
    cfg = {'d1': {'q_low': 0, 'q_up': 1, 'base_n': 1, 'smooth_window': 2}}
    out = fe.add_up_down_targets(df, cfg)
    vals = out['future_max_drawdown_d1'].tolist()
    expected = [-0.3333333333333333, -0.3333333333333333, -0.5]
    assert np.allclose(vals, expected, equal_nan=True)

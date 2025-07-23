import numpy as np
import pandas as pd
from scipy import stats
from quant_trade.feature_engineering import FeatureEngineer


def _make_df():
    times = pd.date_range('2020-01-01', periods=3, freq='h')
    return pd.DataFrame({'open_time': times, 'close': [1.0, 2.0, 3.0], 'symbol': ['BTC']*3})


def test_log_transform():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.rise_transform = 'log'
    fe.boxcox_lambda = {}
    out = fe.add_up_down_targets(_make_df(), {'1h': {'q_low': 0, 'q_up': 1, 'base_n': 1}})
    vals = out['future_max_rise_1h'].dropna().to_numpy()
    assert np.allclose(vals, np.log1p([1.0, 0.5]))


def test_boxcox_transform():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.rise_transform = 'boxcox'
    fe.boxcox_lambda = {}
    out = fe.add_up_down_targets(_make_df(), {'1h': {'q_low': 0, 'q_up': 1, 'base_n': 1}})
    vals = out['future_max_rise_1h'].dropna().to_numpy()
    lmbda = fe.boxcox_lambda['1h']
    expected = stats.boxcox([2.0, 1.5], lmbda)
    assert np.allclose(vals, expected)

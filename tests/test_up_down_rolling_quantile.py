import numpy as np
import pandas as pd
from quant_trade.feature_engineering import FeatureEngineer


def test_add_up_down_targets_rolling_quantile():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe.rise_transform = 'none'
    fe.boxcox_lambda = {}
    df = pd.DataFrame({
        'open_time': pd.date_range('2020-01-01', periods=6, freq='H'),
        'close': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        'symbol': ['BTC'] * 6,
    })
    cfg = {'1h': {'q_low': 0.25, 'q_up': 0.75, 'base_n': 1, 'vol_window': 3, 'min_periods': 1}}
    out = fe.add_up_down_targets(df, cfg)
    expected = [1.0, 0.0, 2.0, 0.0, 2.0, np.nan]
    assert np.allclose(out['target_1h'].to_numpy(), expected, equal_nan=True)

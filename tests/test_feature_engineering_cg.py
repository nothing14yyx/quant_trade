import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import pytest
from utils.helper import calc_features_raw


def test_calc_features_raw_with_cg():
    times = pd.date_range('2020-01-01', periods=3, freq='h')
    df = pd.DataFrame({
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [100, 100, 100],
        'cg_price': [1.4, 1.6, 1.8],
        'cg_market_cap': [1000, 1100, 1200],
        'cg_total_volume': [2000, 2200, 2400],
    }, index=times)

    feats = calc_features_raw(df, '1h')

    assert 'price_diff_cg_1h' in feats
    assert feats['price_diff_cg_1h'].iloc[0] == pytest.approx(0.1)

    assert 'price_ratio_cg_1h' in feats
    assert feats['price_ratio_cg_1h'].iloc[1] == pytest.approx(2.5 / 1.6)

    assert 'cg_market_cap_roc_1h' in feats
    assert feats['cg_market_cap_roc_1h'].iloc[1] == pytest.approx(0.1)

    assert 'cg_total_volume_roc_1h' in feats
    assert feats['cg_total_volume_roc_1h'].iloc[2] == pytest.approx((2400 - 2200) / 2200)

    assert 'volume_cg_ratio_1h' in feats
    assert feats['volume_cg_ratio_1h'].iloc[0] == pytest.approx(100 / 2000)


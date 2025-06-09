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
        'taker_buy_base': [60, 50, 40],
        'btc_close': [1.4, 1.5, 1.6],
        'eth_close': [0.9, 1.0, 1.1],
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

    new_cols = [
        'hv_7d_1h',
        'kc_width_pct_chg_1h',
        'ichimoku_base_1h',
        'buy_sell_ratio_1h',
        'vol_profile_density_1h',
        'money_flow_ratio_1h',
        'skewness_1h',
        'kurtosis_1h',
        'btc_correlation_1h_1h',
        'eth_correlation_1h_1h',
        'bid_ask_spread_pct_1h',
    ]
    for col in new_cols:
        assert col in feats


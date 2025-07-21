import pandas as pd
import pytest

from quant_trade.utils.helper import calc_features_raw
from quant_trade.feature_engineering import calc_cross_features


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
        'AdrActCnt': [100, 110, 121],
        'AdrNewCnt': [50, 60, 55],
        'TxCnt': [10, 12, 12],
        'CapMrktCurUSD': [1000, 1100, 1200],
        'CapRealUSD': [800, 900, 1000],
        'FeeTotUSD': [1.0, 1.1, 0.9],
        'RevHashUSD': [10.0, 11.0, 9.0],
        'IssTotUSD': [5.0, 4.0, 6.0],
        'SplyCur': [100.0, 101.0, 102.0],
        'SplyAct1Yr': [80.0, 81.0, 82.0],
        'HashRate': [50.0, 55.0, 60.0],
        'DiffMean': [10.0, 11.0, 10.0],
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

    assert 'active_addr_roc_1h' in feats
    assert feats['active_addr_roc_1h'].iloc[1] == pytest.approx(0.1)

    assert 'new_addr_roc_1h' in feats
    assert feats['new_addr_roc_1h'].iloc[2] == pytest.approx((55 - 60) / 60)

    assert 'tx_count_roc_1h' in feats
    assert feats['tx_count_roc_1h'].iloc[1] == pytest.approx(0.2)

    assert 'mvrv_ratio_1h' in feats
    assert feats['mvrv_ratio_1h'].iloc[0] == pytest.approx(1000 / 800)

    assert 'fee_tot_roc_1h' in feats
    assert feats['fee_tot_roc_1h'].iloc[1] == pytest.approx(0.1)
    assert 'rev_hash_roc_1h' in feats
    assert feats['rev_hash_roc_1h'].iloc[1] == pytest.approx(0.1)
    assert 'iss_tot_roc_1h' in feats
    assert feats['iss_tot_roc_1h'].iloc[1] == pytest.approx(-0.2)
    assert 'fee_rev_ratio_1h' in feats
    assert feats['fee_rev_ratio_1h'].iloc[0] == pytest.approx(0.1)
    assert 'fee_iss_ratio_1h' in feats
    assert feats['fee_iss_ratio_1h'].iloc[0] == pytest.approx(0.2)
    assert 'iss_rev_ratio_1h' in feats
    assert feats['iss_rev_ratio_1h'].iloc[0] == pytest.approx(0.5)
    assert 'sply_cur_roc_1h' in feats
    assert feats['sply_cur_roc_1h'].iloc[1] == pytest.approx(0.01)
    assert 'sply_act_1yr_roc_1h' in feats
    assert feats['sply_act_1yr_roc_1h'].iloc[1] == pytest.approx(0.0125)
    assert 'sply_act_pct_1h' in feats
    assert feats['sply_act_pct_1h'].iloc[0] == pytest.approx(0.8)
    assert 'hash_rate_roc_1h' in feats
    assert feats['hash_rate_roc_1h'].iloc[1] == pytest.approx(0.1)
    assert 'diff_mean_roc_1h' in feats
    assert feats['diff_mean_roc_1h'].iloc[1] == pytest.approx(0.1)
    assert 'hashrate_difficulty_ratio_1h' in feats
    assert feats['hashrate_difficulty_ratio_1h'].iloc[0] == pytest.approx(5.0)

    new_cols = [
        'hv_7d_1h',
        'kc_width_pct_chg_1h',
        'ichimoku_base_1h',
        'buy_sell_ratio_1h',
        'vol_profile_density_1h',
        'money_flow_ratio_1h',
        'vwap_1h',
        'stoch_k_1h',
        'stoch_d_1h',
        'skewness_1h',
        'kurtosis_1h',
        'btc_correlation_1h_1h',
        'eth_correlation_1h_1h',
        'bid_ask_spread_pct_1h',
        'channel_pos_1h',
    ]
    for col in new_cols:
        assert col in feats


def test_calc_features_raw_minutes():
    times = pd.date_range('2020-01-01', periods=5, freq='5min')
    df = pd.DataFrame({
        'open': [1]*5,
        'high': [1]*5,
        'low': [1]*5,
        'close': [1, 2, 3, 4, 5],
        'volume': [1]*5,
    }, index=times)
    feats5 = calc_features_raw(df, '5m')
    assert 'pct_chg1_5m' in feats5

    times15 = pd.date_range('2020-01-01', periods=5, freq='15min')
    df15 = pd.DataFrame({
        'open': [1]*5,
        'high': [1]*5,
        'low': [1]*5,
        'close': [1, 2, 3, 4, 5],
        'volume': [1]*5,
    }, index=times15)
    feats15 = calc_features_raw(df15, '15m')
    assert 'pct_chg1_15m' in feats15


def test_sma_and_ma_ratio():
    times = pd.date_range('2020-01-01', periods=20, freq='h')
    close = pd.Series(range(1, 21), index=times)
    df = pd.DataFrame({
        'open': close,
        'high': close,
        'low': close,
        'close': close,
        'volume': [1]*20,
    }, index=times)

    feats = calc_features_raw(df, '1h')
    assert 'sma_5_1h' in feats
    assert 'sma_20_1h' in feats

    q = close.quantile([0.001, 0.999])
    clipped = close.clip(q.loc[0.001], q.loc[0.999])
    expected_sma5 = clipped.rolling(5).mean().iloc[-1]
    expected_sma20 = clipped.rolling(20).mean().iloc[-1]
    assert feats['sma_5_1h'].iloc[-1] == pytest.approx(expected_sma5)
    assert feats['sma_20_1h'].iloc[-1] == pytest.approx(expected_sma20)

    f1h = feats.copy()
    f4h = calc_features_raw(df, '4h')
    f1d = calc_features_raw(df, 'd1')
    merged = calc_cross_features(f1h, f4h, f1d)

    ratio = expected_sma5 / expected_sma20
    assert 'ma_ratio_5_20' in merged
    assert merged['ma_ratio_5_20'].iloc[-1] == pytest.approx(ratio)

    # channel position should be 1.0 for increasing series
    assert feats['channel_pos_1h'].iloc[-1] == pytest.approx(1.0)


def test_calc_features_raw_index_sort_unique():
    times = pd.to_datetime([
        '2020-01-02 01:00',
        '2020-01-01 00:00',
        '2020-01-02 01:00',
    ])
    df = pd.DataFrame(
        {
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3],
            'close': [1, 2, 3],
            'volume': [1, 1, 1],
        },
        index=times,
    )

    feats = calc_features_raw(df, '1h')

    assert feats.index.is_monotonic_increasing
    assert not feats.index.has_duplicates


def test_onchain_features_cross_merge():
    times = pd.date_range('2020-01-01', periods=3, freq='h')
    df = pd.DataFrame({
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [100, 100, 100],
        'AdrActCnt': [100, 110, 121],
        'AdrNewCnt': [50, 60, 55],
        'TxCnt': [10, 12, 12],
        'CapMrktCurUSD': [1000, 1100, 1200],
        'CapRealUSD': [800, 900, 1000],
    }, index=times)

    f1h = calc_features_raw(df, '1h')
    f4h = calc_features_raw(df, '4h')
    f1d = calc_features_raw(df, 'd1')
    merged = calc_cross_features(f1h, f4h, f1d)

    assert 'active_addr_roc_4h' in merged
    assert 'mvrv_ratio_d1' in merged


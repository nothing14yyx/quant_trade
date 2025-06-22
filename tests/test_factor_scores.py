import pytest
from collections import deque

from robust_signal_generator import RobustSignalGenerator


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg.base_weights = {
        'ai': 0.2,
        'trend': 0.2,
        'momentum': 0.2,
        'volatility': 0.2,
        'volume': 0.1,
        'sentiment': 0.05,
        'funding': 0.05,
    }
    rsg.ic_scores = {k: 1 for k in rsg.base_weights}
    rsg.current_weights = rsg.base_weights.copy()
    rsg.history_scores = deque(maxlen=500)
    rsg.oi_change_history = deque(maxlen=500)
    rsg.ic_history = {k: deque(maxlen=500) for k in rsg.base_weights}
    rsg.symbol_categories = {}
    rsg._prev_raw = {p: None for p in ("1h", "4h", "d1")}
    return rsg


def test_new_features_affect_scores():
    rsg = make_rsg()
    base_feats = {
        'ema_diff_1h': 0,
        'boll_perc_1h': 0.5,
        'supertrend_dir_1h': 1,
        'adx_delta_1h': 0,
        'bull_streak_1h': 1,
        'bear_streak_1h': 0,
        'long_lower_shadow_1h': 0,
        'long_upper_shadow_1h': 0,
        'vol_breakout_1h': 0,
        'rsi_1h': 50,
        'willr_1h': -50,
        'macd_hist_1h': 0,
        'rsi_slope_1h': 0,
        'mfi_1h': 50,
        'atr_pct_1h': 0.01,
        'bb_width_1h': 0.02,
        'donchian_delta_1h': 0.01,
        'hv_7d_1h': 0.01,
        'hv_14d_1h': 0.01,
        'hv_30d_1h': 0.01,
        'vol_ma_ratio_1h': 1,
        'obv_delta_1h': 0,
        'vol_roc_1h': 0,
        'rsi_mul_vol_ma_ratio_1h': 50,
        'buy_sell_ratio_1h': 1,
        'vol_profile_density_1h': 5,
        'fg_index_d1': 50,
        'btc_correlation_1h_1h': 0,
        'eth_correlation_1h_1h': 0,
        'price_diff_cg_1h': 0,
        'cg_market_cap_roc_1h': 0,
        'volume_cg_ratio_1h': 1,
        'funding_rate_1h': 0,
        'funding_rate_anom_1h': 0,
        'ichimoku_cloud_thickness_1h': 0.01,
        'vwap_1h': 100,
        'close': 100,
        'rsi_diff_1h_4h': 0,
        'rsi_diff_1h_d1': 0,
        'rsi_diff_4h_d1': 0,
    }

    base = rsg.get_factor_scores(base_feats, '1h')

    thicker = base_feats.copy()
    thicker['ichimoku_cloud_thickness_1h'] = 0.05
    assert rsg.get_factor_scores(thicker, '1h')['trend'] > base['trend']

    higher_diff = base_feats.copy()
    higher_diff['rsi_diff_1h_4h'] = 5
    assert rsg.get_factor_scores(higher_diff, '1h')['momentum'] > base['momentum']

    higher_vwap = base_feats.copy()
    higher_vwap['close'] = 105
    assert rsg.get_factor_scores(higher_vwap, '1h')['trend'] > base['trend']



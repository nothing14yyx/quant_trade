import pytest
from collections import deque

from quant_trade.robust_signal_generator import RobustSignalGenerator


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
    rsg.regime_adx_trend = 25
    rsg.regime_adx_range = 20
    rsg.risk_adjust_factor = 0.9
    rsg.risk_adjust_threshold = 0
    rsg.risk_score_limit = 2.0
    rsg.crowding_limit = 1.1
    rsg.max_position = 0.3
    rsg.risk_scale = 1.0
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
        'kc_perc_1h': 0.5,
        'kc_width_pct_chg_1h': 0,
        'donchian_perc_1h': 0.5,
        'money_flow_ratio_1h': 1,
        'skewness_1h': 0,
        'kurtosis_1h': 0,
        'bid_ask_spread_pct_1h': 0,
        'sma_10_1h': 100,
        'sma_20_1h': 100,
        'stoch_k_1h': 50,
        'stoch_d_1h': 50,
        'macd_signal_1h': 0,
        'pct_chg1_1h': 0,
        'pct_chg3_1h': 0,
        'pct_chg6_1h': 0,
        'cci_1h': 0,
        'cci_delta_1h': 0,
        'atr_chg_1h': 0,
        'bb_width_chg_1h': 0,
        'vol_ma_ratio_long_1h': 1,
        'cg_total_volume_roc_1h': 0,
        'price_ratio_cg_1h': 1,
        'bid_ask_imbalance': 0,
        'ichimoku_base_1h': 100,
        'ichimoku_conversion_1h': 100,
        'close': 100,
        'rsi_diff_1h_4h': 0,
        'rsi_diff_1h_d1': 0,
        'rsi_diff_4h_d1': 0,
        'channel_pos_1h': 0.5,
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


    high_kc = base_feats.copy()
    high_kc['kc_perc_1h'] = 0.8
    assert rsg.get_factor_scores(high_kc, '1h')['trend'] > base['trend']

    wide_vol = base_feats.copy()
    wide_vol['kc_width_pct_chg_1h'] = 0.5
    assert rsg.get_factor_scores(wide_vol, '1h')['volatility'] > base['volatility']

    strong_flow = base_feats.copy()
    strong_flow['money_flow_ratio_1h'] = 3
    assert rsg.get_factor_scores(strong_flow, '1h')['volume'] > base['volume']

    near_top = base_feats.copy()
    near_top['channel_pos_1h'] = 0.95
    assert rsg.get_factor_scores(near_top, '1h')['trend'] < base['trend']

    breakout_up = base_feats.copy()
    breakout_up['channel_pos_1h'] = 1.05
    assert rsg.get_factor_scores(breakout_up, '1h')['trend'] > base['trend']

    stoch_up = base_feats.copy()
    stoch_up['stoch_k_1h'] = 80
    assert rsg.get_factor_scores(stoch_up, '1h')['momentum'] > base['momentum']

    pct_up = base_feats.copy()
    pct_up['pct_chg1_1h'] = 0.03
    assert rsg.get_factor_scores(pct_up, '1h')['momentum'] > base['momentum']

    atr_jump = base_feats.copy()
    atr_jump['atr_chg_1h'] = 0.01
    assert rsg.get_factor_scores(atr_jump, '1h')['volatility'] > base['volatility']

    vol_long = base_feats.copy()
    vol_long['vol_ma_ratio_long_1h'] = 2
    assert rsg.get_factor_scores(vol_long, '1h')['volume'] > base['volume']

    price_ratio = base_feats.copy()
    price_ratio['price_ratio_cg_1h'] = 1.1
    assert rsg.get_factor_scores(price_ratio, '1h')['sentiment'] > base['sentiment']

    imbalance = base_feats.copy()
    imbalance['bid_ask_imbalance'] = 0.3
    assert rsg.get_factor_scores(imbalance, '1h')['volume'] > base['volume']

    spread = base_feats.copy()
    spread['close_spread_1h_4h'] = 0.05
    assert rsg.get_factor_scores(spread, '1h')['trend'] > base['trend']

    macd_mult = base_feats.copy()
    macd_mult['macd_hist_4h_mul_bb_width_1h'] = 0.02
    assert rsg.get_factor_scores(macd_mult, '1h')['momentum'] > base['momentum']

    mom_std = base_feats.copy()
    mom_std['mom_5m_roll1h_std'] = 0.1
    assert rsg.get_factor_scores(mom_std, '1h')['volatility'] > base['volatility']

    vol_ratio = base_feats.copy()
    vol_ratio['vol_ratio_4h_d1'] = 1.2
    assert rsg.get_factor_scores(vol_ratio, '1h')['volume'] > base['volume']

    vol_rsi = base_feats.copy()
    vol_rsi['rsi_1h_mul_vol_ma_ratio_4h'] = 10
    scores = rsg.get_factor_scores(vol_rsi, '1h')
    assert scores['volume'] > base['volume']
    assert scores['momentum'] == base['momentum']


def test_bb_squeeze_reduces_volatility():
    rsg = make_rsg()
    base_feats = {
        'atr_pct_1h': 0.01,
        'bb_width_1h': 0.02,
        'bb_width_chg_1h': 0,
        'bb_squeeze_1h': 0,
    }
    base_score = rsg.get_factor_scores(base_feats, '1h')['volatility']
    squeezed = base_feats.copy()
    squeezed['bb_squeeze_1h'] = 1
    squeeze_score = rsg.get_factor_scores(squeezed, '1h')['volatility']
    assert squeeze_score < base_score


def test_td_score_changes_trend():
    rsg = make_rsg()
    base_feats = {
        'td_buy_count_1h': 0,
        'td_sell_count_1h': 0,
    }
    base_score = rsg.get_factor_scores(base_feats, '1h')['trend']

    sell_more = base_feats.copy()
    sell_more['td_sell_count_1h'] = 9
    assert rsg.get_factor_scores(sell_more, '1h')['trend'] > base_score

    buy_more = base_feats.copy()
    buy_more['td_buy_count_1h'] = 9
    assert rsg.get_factor_scores(buy_more, '1h')['trend'] < base_score




import math
from collections import deque, OrderedDict

from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal.factor_scorer import FactorScorerImpl


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = OrderedDict()
    rsg.factor_scorer = FactorScorerImpl(rsg)
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
    rsg.volume_quantile_low = 0.2
    rsg.volume_quantile_high = 0.8
    rsg.volume_ratio_history = deque([0.8, 1.0, 1.2], maxlen=500)
    return rsg


def test_trend_score_basic():
    rsg = make_rsg()
    base = {'price_vs_ma200_1h': 0, 'ema_slope_50_1h': 0, 'adx_1h': 0}
    higher = {'price_vs_ma200_1h': 0.05, 'ema_slope_50_1h': 0.1, 'adx_1h': 30}
    base_score = rsg.factor_scorer.score(base, '1h')['trend']
    higher_score = rsg.factor_scorer.score(higher, '1h')['trend']
    assert higher_score > base_score


def test_momentum_score_basic():
    rsg = make_rsg()
    low = {'rsi_1h': 40, 'macd_hist_1h': 0}
    high = {'rsi_1h': 60, 'macd_hist_1h': 0.02}
    assert rsg.factor_scorer.score(high, '1h')['momentum'] > rsg.factor_scorer.score(low, '1h')['momentum']


def test_volatility_score_basic():
    rsg = make_rsg()
    low = {'atr_pct_1h': 0.01, 'bb_width_1h': 0.02}
    high = {'atr_pct_1h': 0.05, 'bb_width_1h': 0.04}
    assert rsg.factor_scorer.score(high, '1h')['volatility'] > rsg.factor_scorer.score(low, '1h')['volatility']


def test_volume_score_basic():
    rsg = make_rsg()
    low = {'vol_ma_ratio_1h': 0.5, 'buy_sell_ratio_1h': 1.0}
    high = {'vol_ma_ratio_1h': 2.0, 'buy_sell_ratio_1h': 1.2}
    assert rsg.factor_scorer.score(high, '1h')['volume'] > rsg.factor_scorer.score(low, '1h')['volume']


def test_sentiment_score_basic():
    rsg = make_rsg()
    negative = {'funding_rate_1h': -0.01}
    positive = {'funding_rate_1h': 0.01}
    assert rsg.factor_scorer.score(positive, '1h')['sentiment'] > rsg.factor_scorer.score(negative, '1h')['sentiment']

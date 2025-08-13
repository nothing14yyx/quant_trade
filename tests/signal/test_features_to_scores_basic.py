from quant_trade.utils.lru import LRU
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal.features_to_scores import get_factor_scores


def make_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = LRU(300)
    return rsg


def test_trend_score_basic():
    rsg = make_rsg()
    base = {"price_vs_ma200_1h": 0, "ema_slope_50_1h": 0, "adx_1h": 0}
    higher = {"price_vs_ma200_1h": 0.05, "ema_slope_50_1h": 0.1, "adx_1h": 30}
    base_score = get_factor_scores(rsg, base, "1h")["trend"]
    higher_score = get_factor_scores(rsg, higher, "1h")["trend"]
    assert higher_score > base_score


def test_momentum_score_basic():
    rsg = make_rsg()
    low = {"rsi_1h": 40, "macd_hist_1h": 0}
    high = {"rsi_1h": 60, "macd_hist_1h": 0.02}
    assert get_factor_scores(rsg, high, "1h")["momentum"] > get_factor_scores(
        rsg, low, "1h"
    )["momentum"]

import pandas as pd
from quant_trade.market_phase import MarketPhaseCalculator


def test_rolling_zscore():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    z = MarketPhaseCalculator.rolling_zscore(series, window=3)
    expected = (series - series.rolling(window=3, min_periods=1).mean()) / series.rolling(window=3, min_periods=1).std()
    expected = expected.replace([float("inf"), float("-inf")], pd.NA)
    pd.testing.assert_series_equal(z, expected)


def test_normalize_weights():
    metrics = ["A", "B", "C"]
    weights = {"A": 2.0, "B": 1.0}
    norm = MarketPhaseCalculator.normalize_weights(weights, metrics)
    expected = pd.Series([2.0, 1.0, 1.0], index=metrics) / 4.0
    pd.testing.assert_series_equal(norm, expected)

    # zero sum
    norm2 = MarketPhaseCalculator.normalize_weights({"A": 0, "B": 0, "C": 0}, metrics)
    expected2 = pd.Series([1/3, 1/3, 1/3], index=metrics)
    pd.testing.assert_series_equal(norm2, expected2)


def test_phase_from_score():
    assert MarketPhaseCalculator.phase_from_score(None) == "range"
    assert MarketPhaseCalculator.phase_from_score(0.0) == "range"
    assert MarketPhaseCalculator.phase_from_score(0.1) == "bull"
    assert MarketPhaseCalculator.phase_from_score(-0.1) == "bear"

from quant_trade.backtest.backtester import execute_signal


def test_execute_signal_unfilled_buy():
    # Order placed slightly below the next open price
    price_open_next = 100.0
    price_high_next = 101.0
    price_low_next = 99.6  # price never drops enough to fill
    fill, pnl = execute_signal(
        price_open_next,
        price_high_next,
        price_low_next,
        1,
        slippage=0.01,
    )
    assert fill is None
    assert pnl == 0.0

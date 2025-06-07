import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from backtester import simulate_trades


def test_simulate_trades_tp_hit():
    times = pd.date_range('2020-01-01', periods=4, freq='H')
    df_sym = pd.DataFrame({
        'symbol': ['BTC'] * 4,
        'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'open': [100, 100, 104, 106],
        'high': [100, 106, 107, 108],
        'low': [100, 99, 103, 105],
        'close': [100, 104, 106, 107],
    })

    sig_df = pd.DataFrame({
        'open_time': [times[1]],
        'signal': [1],
        'score': [0.5],
        'position_size': [1.0],
        'take_profit': [107],
        'stop_loss': [95],
    })

    trades = simulate_trades(df_sym, sig_df, fee_rate=0, slippage=0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t['entry_time'] == times[1]
    assert t['exit_time'] == times[2]
    assert t['exit_price'] == 107
    assert t['pnl'] == 7

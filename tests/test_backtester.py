import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from backtester import simulate_trades


def test_simulate_trades_tp_hit():
    times = pd.date_range('2020-01-01', periods=4, freq='h')
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

def test_simulate_trades_sl_hit():
    times = pd.date_range('2020-01-01', periods=4, freq='h')
    df_sym = pd.DataFrame({
        'symbol': ['BTC'] * 4,
        'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'open': [100, 100, 100, 100],
        'high': [100, 101, 102, 102],
        'low': [100, 99, 94, 96],
        'close': [100, 100, 95, 96],
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
    assert t['exit_price'] == 95
    assert t['pnl'] == -5


def test_simulate_trades_reverse_signal_exit():
    times = pd.date_range('2020-01-01', periods=4, freq='h')
    df_sym = pd.DataFrame({
        'symbol': ['BTC'] * 4,
        'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'open': [100, 100, 100, 100],
        'high': [100, 104, 104, 104],
        'low': [100, 96, 96, 96],
        'close': [100, 100, 100, 101],
    })

    sig_df = pd.DataFrame({
        'open_time': [times[1], times[2], times[3], times[3]],
        'signal': [1, 0, 0, -1],
        'score': [0.5, 0, 0, 0],
        'position_size': [1.0, 1.0, 1.0, 1.0],
        'take_profit': [107, 107, 107, 107],
        'stop_loss': [95, 95, 95, 95],
    })

    trades = simulate_trades(df_sym, sig_df, fee_rate=0, slippage=0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t['entry_time'] == times[1]
    assert t['exit_time'] == times[3] + pd.Timedelta(hours=1)
    assert t['exit_price'] == 101

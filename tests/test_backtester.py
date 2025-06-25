import pandas as pd
import pytest

from quant_trade.backtester import simulate_trades


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


def test_ret_and_win_rate_with_position_sizes():
    times = pd.date_range('2020-01-01', periods=6, freq='h')
    df_sym = pd.DataFrame({
        'symbol': ['BTC'] * 6,
        'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'open': [100, 100, 102, 103, 104, 99],
        'high': [100, 103, 103, 105, 107, 100],
        'low': [100, 99, 101, 102, 102, 98],
        'close': [100, 102, 103, 104, 99, 100],
    })

    sig_df = pd.DataFrame({
        'open_time': times[:-1],
        'signal': [1, 0, -1, 0, 1],
        'score': [0.5, 0.0, 0.5, 0.0, 0.5],
        'position_size': [0.5, 0.0, 1.0, 0.0, 0.0],
        'take_profit': [103, 0, 98, 0, 110],
        'stop_loss': [90, 0, 106, 0, 90],
    })

    trades = simulate_trades(df_sym, sig_df, fee_rate=0, slippage=0)
    assert len(trades) == 2
    assert trades['ret'].tolist() == [
        pytest.approx(0.03),
        pytest.approx(-3 / 103),
    ]

    weights = trades['position_size']
    win_mask = trades['ret'] > 0
    win_rate = weights[win_mask].sum() / weights.sum()
    assert win_rate == pytest.approx(1 / 3)

    series = trades['ret'] * trades['position_size']
    total_ret = (series + 1.0).cumprod().iloc[-1] - 1.0
    assert total_ret == pytest.approx(-0.014563, rel=1e-4)


def test_total_ret_all_in():
    times = pd.date_range('2020-01-01', periods=6, freq='h')
    df_sym = pd.DataFrame({
        'symbol': ['BTC'] * 6,
        'open_time': times,
        'close_time': times + pd.Timedelta(hours=1),
        'open': [100, 100, 102, 103, 104, 99],
        'high': [100, 103, 103, 105, 107, 100],
        'low': [100, 99, 101, 102, 102, 98],
        'close': [100, 102, 103, 104, 99, 100],
    })

    sig_df = pd.DataFrame({
        'open_time': times[:-1],
        'signal': [1, 0, -1, 0, 1],
        'score': [0.5, 0.0, 0.5, 0.0, 0.5],
        'position_size': [1.0] * 5,
        'take_profit': [103, 0, 98, 0, 110],
        'stop_loss': [90, 0, 106, 0, 90],
    })

    trades = simulate_trades(df_sym, sig_df, fee_rate=0, slippage=0)
    series = trades['ret'] * trades['position_size']
    total_ret = (series + 1.0).cumprod().iloc[-1] - 1.0
    assert total_ret == pytest.approx(0.0, rel=1e-4)

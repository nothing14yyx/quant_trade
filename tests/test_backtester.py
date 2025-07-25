import pandas as pd
import numpy as np
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


def test_run_backtest_recent_days(monkeypatch):
    import quant_trade.backtester as bt

    monkeypatch.setattr(bt, "load_config", lambda: {})
    monkeypatch.setattr(bt, "connect_mysql", lambda cfg: None)

    queries: list[tuple[str, dict | None]] = []

    def fake_read_sql(sql, engine, parse_dates=None, params=None):
        queries.append((sql, params))
        if "MAX(open_time" in sql:
            return pd.DataFrame({"end_time": [pd.Timestamp("2024-01-05")]} )
        assert "WHERE open_time >= %(start)s" in sql
        assert params["start"] == pd.Timestamp("2024-01-03")
        df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 3,
                "open_time": pd.to_datetime([
                    "2024-01-05 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                ]),
                "close_time": pd.to_datetime([
                    "2024-01-05 01:00:00",
                    "2024-01-03 01:00:00",
                    "2024-01-04 01:00:00",
                ]),
                "open": [1, 1, 1],
                "high": [1, 1, 1],
                "low": [1, 1, 1],
                "close": [1, 1, 1],
                "volume": [1, 1, 1],
            }
        )
        return df

    monkeypatch.setattr(bt.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(bt, "calc_features_raw", lambda df, period: pd.DataFrame(index=df.index))
    monkeypatch.setattr(bt, "simulate_trades", lambda *a, **k: pd.DataFrame(columns=["position_size", "ret"]))
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)
    monkeypatch.setattr(bt, "FEATURE_COLS_1H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_4H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_D1", [])

    captured = {}

    class DummyRSG:
        def __init__(self, *a, **k):
            pass

        def update_ic_scores(self, df, group_by=None):
            captured["times"] = df["open_time"].copy()
            raise StopIteration

        def generate_signal(self, *a, **k):
            return {
                "signal": 0,
                "score": 0.0,
                "position_size": 0.0,
                "take_profit": None,
                "stop_loss": None,
            }

    class DummyCfg:
        @staticmethod
        def from_cfg(cfg):
            return None

    monkeypatch.setattr(bt, "RobustSignalGenerator", DummyRSG)
    monkeypatch.setattr(bt, "RobustSignalGeneratorConfig", DummyCfg)

    with pytest.raises(StopIteration):
        bt.run_backtest(recent_days=2)

    assert any("MAX(open_time" in q for q, _ in queries)
    assert any("WHERE open_time >= %(start)s" in q for q, _ in queries)
    times = captured["times"]
    assert times.is_monotonic_increasing
    assert times.min() >= pd.Timestamp("2024-01-03")


def test_run_backtest_skip_invalid_features(monkeypatch, caplog):
    import logging
    import quant_trade.backtester as bt

    monkeypatch.setattr(bt, "load_config", lambda: {})
    monkeypatch.setattr(bt, "connect_mysql", lambda cfg: None)

    def fake_read_sql(sql, engine, parse_dates=None, params=None):
        return pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 3,
                "open_time": pd.date_range("2024-01-01", periods=3, freq="h"),
                "close_time": pd.date_range("2024-01-01", periods=3, freq="h")
                + pd.Timedelta(hours=1),
                "open": [1, 1, 1],
                "high": [1, 1, 1],
                "low": [1, 1, 1],
                "close": [1, 1, 1],
                "volume": [1, 1, 1],
                "feat": [np.nan, np.nan, np.nan],
            }
        )

    monkeypatch.setattr(bt.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(bt, "calc_features_raw", lambda df, period: pd.DataFrame(index=df.index))
    monkeypatch.setattr(bt, "simulate_trades", lambda *a, **k: pd.DataFrame(columns=["position_size", "ret"]))
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *a, **k: None)
    monkeypatch.setattr(bt, "FEATURE_COLS_1H", ["feat"])
    monkeypatch.setattr(bt, "FEATURE_COLS_4H", [])
    monkeypatch.setattr(bt, "FEATURE_COLS_D1", [])

    class DummyRSG:
        def __init__(self, *a, **k):
            pass

        def update_ic_scores(self, df, group_by=None):
            pass

        def generate_signal(self, *a, **k):
            return {
                "signal": 0,
                "score": 0.0,
                "position_size": 0.0,
                "take_profit": None,
                "stop_loss": None,
            }

    class DummyCfg:
        @staticmethod
        def from_cfg(cfg):
            return None

    monkeypatch.setattr(bt, "RobustSignalGenerator", DummyRSG)
    monkeypatch.setattr(bt, "RobustSignalGeneratorConfig", DummyCfg)

    with caplog.at_level(logging.WARNING):
        bt.run_backtest()

    assert "缺少有效特征列" in caplog.text

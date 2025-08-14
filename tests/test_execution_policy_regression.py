import pandas as pd
import numpy as np

from quant_trade.backtester import simulate_trades, calc_equity_curve


def simulate_trades_old(df_sym: pd.DataFrame, sig_df: pd.DataFrame, *, fee_rate: float, slippage: float) -> pd.DataFrame:
    trades = []
    in_pos = False
    entry_price = entry_time = pos_size = score = direction = tp = sl = None
    for i in range(1, len(df_sym) - 1):
        if not in_pos:
            if (
                i - 1 < len(sig_df)
                and sig_df.at[i - 1, "signal"] != 0
                and sig_df.at[i - 1, "position_size"] > 0
                and not np.isnan(sig_df.at[i - 1, "position_size"])
            ):
                direction = sig_df.at[i - 1, "signal"]
                entry_price = df_sym.at[i, "open"] * (1 + slippage * direction)
                entry_time = df_sym.at[i, "open_time"]
                pos_size = sig_df.at[i - 1, "position_size"]
                score = sig_df.at[i - 1, "score"]
                tp = sig_df.at[i - 1, "take_profit"]
                sl = sig_df.at[i - 1, "stop_loss"]
                in_pos = True
            continue

        high = df_sym.at[i, "high"]
        low = df_sym.at[i, "low"]
        exit_price = None
        exit_time = None
        if direction == 1:
            if sl is not None and low <= sl:
                exit_price = sl
                exit_time = df_sym.at[i, "open_time"]
            elif tp is not None and high >= tp:
                exit_price = tp
                exit_time = df_sym.at[i, "open_time"]
        else:
            if sl is not None and high >= sl:
                exit_price = sl
                exit_time = df_sym.at[i, "open_time"]
            elif tp is not None and low <= tp:
                exit_price = tp
                exit_time = df_sym.at[i, "open_time"]

        if exit_price is None and i < len(sig_df) and sig_df.at[i, "signal"] == -direction:
            exit_price = df_sym.at[i + 1, "open"] * (1 - slippage * direction)
            exit_time = df_sym.at[i + 1, "open_time"]

        if exit_price is not None:
            pnl = (exit_price - entry_price) * direction * pos_size
            if pos_size:
                ret = pnl / (entry_price * pos_size) - 2 * fee_rate
            else:
                ret = 0.0
            holding_s = (exit_time - entry_time).total_seconds()
            trades.append(
                {
                    "symbol": df_sym.at[i, "symbol"],
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "signal": direction,
                    "score": score,
                    "position_size": pos_size,
                    "pnl": pnl,
                    "ret": ret,
                    "holding_s": holding_s,
                }
            )
            in_pos = False

    if in_pos:
        exit_price = df_sym.at[len(df_sym) - 1, "close"] * (1 - slippage * direction)
        exit_time = df_sym.at[len(df_sym) - 1, "close_time"]
        pnl = (exit_price - entry_price) * direction * pos_size
        if pos_size:
            ret = pnl / (entry_price * pos_size) - 2 * fee_rate
        else:
            ret = 0.0
        holding_s = (exit_time - entry_time).total_seconds()
        trades.append(
            {
                "symbol": df_sym.at[len(df_sym) - 1, "symbol"],
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "signal": direction,
                "score": score,
                "position_size": pos_size,
                "pnl": pnl,
                "ret": ret,
                "holding_s": holding_s,
            }
        )

    return pd.DataFrame(trades)


def test_execution_policy_changes_fill_rate_and_metrics():
    times = pd.date_range("2020-01-01", periods=5, freq="h")
    df_sym = pd.DataFrame(
        {
            "symbol": ["BTC"] * 5,
            "open_time": times,
            "close_time": times + pd.Timedelta(hours=1),
            "open": [100, 100, 104, 103, 103],
            "high": [101, 105, 104, 104, 103],
            "low": [99, 100, 103, 102, 103],
            "close": [100, 104, 103, 103, 103],
        }
    )
    sig_df = pd.DataFrame(
        {
            "open_time": [times[0], times[1], times[2]],
            "signal": [1, 0, 1],
            "score": [0.9, 0.0, 0.6],
            "position_size": [1.0, 0.0, 1.0],
            "take_profit": [104, 0, 104],
            "stop_loss": [95, 0, 95],
        }
    )

    fee_rate = 0.001
    slippage = 0.01

    trades_old = simulate_trades_old(df_sym, sig_df, fee_rate=fee_rate, slippage=slippage)
    trades_new = simulate_trades(df_sym, sig_df, fee_rate=fee_rate, slippage=slippage)

    sig_mask = (sig_df["signal"] != 0) & (sig_df["position_size"] > 0)
    strong_total = (sig_mask & (sig_df["score"] >= 0.8)).sum()
    weak_total = (sig_mask & (sig_df["score"] < 0.8)).sum()

    strong_rate_old = (trades_old["score"] >= 0.8).sum() / strong_total
    weak_rate_old = (trades_old["score"] < 0.8).sum() / weak_total
    strong_rate_new = (trades_new["score"] >= 0.8).sum() / strong_total
    weak_rate_new = (trades_new["score"] < 0.8).sum() / weak_total

    assert strong_rate_new == strong_rate_old == 1.0
    assert weak_rate_old == 1.0
    assert weak_rate_new < weak_rate_old

    ec_old = calc_equity_curve(trades_old)
    ec_new = calc_equity_curve(trades_new)
    ret_old = ec_old.iloc[-1] - 1
    ret_new = ec_new.iloc[-1] - 1
    dd_old = (ec_old.cummax() - ec_old).max()
    dd_new = (ec_new.cummax() - ec_new).max()

    assert ret_new != ret_old
    assert dd_new != dd_old

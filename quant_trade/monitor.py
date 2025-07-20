import time
import pandas as pd
import altair as alt
from sqlalchemy import text

from quant_trade.utils import load_config, connect_mysql
from quant_trade.utils.db import CONFIG_PATH


def fetch_recent(engine, limit=1000):
    sig_query = text(
        "SELECT open_time, signal, score, oi_change FROM signals ORDER BY open_time DESC LIMIT :lim"
    )
    fac_query = text(
        "SELECT open_time, ai, trend, momentum, volatility, volume, sentiment, funding FROM factor_scores ORDER BY open_time DESC LIMIT :lim"
    )
    sig = pd.read_sql(sig_query, engine, params={"lim": limit})
    fac = pd.read_sql(fac_query, engine, params={"lim": limit})
    sig = sig.sort_values("open_time")
    fac = fac.sort_values("open_time")
    return sig, fac


def plot_ic_curve(df: pd.DataFrame) -> alt.Chart:
    df = df.dropna(subset=["score", "signal"]).copy()
    df["ic"] = df["score"].rolling(50).corr(df["signal"])
    return (
        alt.Chart(df)
        .mark_line()
        .encode(x="open_time:T", y="ic:Q")
        .properties(title="IC Curve")
    )


def plot_oi_change(df: pd.DataFrame) -> alt.Chart:
    if "oi_change" not in df.columns:
        return alt.Chart(pd.DataFrame(columns=["open_time", "oi_change"]))
    return (
        alt.Chart(df)
        .mark_line()
        .encode(x="open_time:T", y="oi_change:Q")
        .properties(title="OI Change")
    )


def run_monitor(cfg_path: str | None = None, interval: int = 3600, limit: int = 1000):
    cfg = load_config(CONFIG_PATH if cfg_path is None else cfg_path)
    engine = connect_mysql(cfg)
    while True:
        sig, fac = fetch_recent(engine, limit=limit)
        ic_chart = plot_ic_curve(sig)
        oi_chart = plot_oi_change(sig)
        ic_chart.save("ic_curve.html")
        oi_chart.save("oi_change.html")
        time.sleep(interval)


__all__ = ["run_monitor", "fetch_recent", "plot_ic_curve", "plot_oi_change"]

import pandas as pd
from sqlalchemy import text
from pathlib import Path

from .config_manager import ConfigManager

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


def detect_market_phase(engine, config_path: str | Path = CONFIG_PATH) -> str:
    """根据活跃地址与市值判断市场阶段。

    返回值为 "bull"、"bear" 或 "range"。
    """
    cfg = ConfigManager(config_path).get("market_phase", {})
    metrics = cfg.get("metrics", ["AdrActCnt", "CapMrktCurUSD", "FeeTotUSD"])
    symbol = cfg.get("symbol", "BTCUSDT")
    base = ["AdrActCnt", "CapMrktCurUSD"]
    metrics = list(dict.fromkeys(base + [m for m in metrics if m not in base]))
    placeholders = ",".join(f"'{m}'" for m in metrics)
    q = text(
        "SELECT timestamp, metric, value FROM cm_onchain_metrics "
        "WHERE symbol=:symbol AND metric IN ({placeholders}) "
        "ORDER BY timestamp DESC LIMIT 120".format(placeholders=placeholders)
    )
    df = pd.read_sql(q, engine, params={"symbol": symbol}, parse_dates=["timestamp"])
    if df.empty:
        return "range"

    df = (
        df.pivot(index="timestamp", columns="metric", values="value")
        .sort_index()
    )

    aa = pd.to_numeric(df.get("AdrActCnt"), errors="coerce")
    cap = pd.to_numeric(df.get("CapMrktCurUSD"), errors="coerce")
    if aa.dropna().empty or cap.dropna().empty:
        return "range"

    aa_ma = aa.rolling(window=30, min_periods=1).mean().iloc[-1]
    cap_ma = cap.rolling(window=30, min_periods=1).mean().iloc[-1]

    cur_aa = aa.iloc[-1]
    cur_cap = cap.iloc[-1]

    bull_cond = cur_aa > aa_ma and cur_cap > cap_ma
    bear_cond = cur_aa < aa_ma and cur_cap < cap_ma

    for m in metrics:
        if m in ("AdrActCnt", "CapMrktCurUSD"):
            continue
        series = pd.to_numeric(df.get(m), errors="coerce")
        if series.dropna().empty:
            continue
        ma = series.rolling(window=30, min_periods=1).mean().iloc[-1]
        cur = series.iloc[-1]
        bull_cond &= cur > ma
        bear_cond &= cur < ma

    if bull_cond:
        return "bull"
    if bear_cond:
        return "bear"
    return "range"

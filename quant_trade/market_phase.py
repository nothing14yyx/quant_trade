import pandas as pd
from sqlalchemy import text


def detect_market_phase(engine) -> str:
    """根据活跃地址与市值判断市场阶段。

    返回值为 "bull"、"bear" 或 "range"。
    """
    q = text(
        "SELECT timestamp, metric, value FROM cm_onchain_metrics "
        "WHERE metric IN ('AdrActCnt','CapMrktCurUSD') "
        "ORDER BY timestamp DESC LIMIT 120"
    )
    df = pd.read_sql(q, engine, parse_dates=["timestamp"])
    if df.empty:
        return "range"

    df = (
        df.pivot(index="timestamp", columns="metric", values="value")
        .sort_index()
    )
    aa = pd.to_numeric(df["AdrActCnt"], errors="coerce")
    cap = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
    aa_ma = aa.rolling(window=30, min_periods=1).mean().iloc[-1]
    cap_ma = cap.rolling(window=30, min_periods=1).mean().iloc[-1]
    cur_aa = aa.iloc[-1]
    cur_cap = cap.iloc[-1]

    if cur_aa > aa_ma and cur_cap > cap_ma:
        return "bull"
    if cur_aa < aa_ma and cur_cap < cap_ma:
        return "bear"
    return "range"

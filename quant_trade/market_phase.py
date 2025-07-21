import pandas as pd
from sqlalchemy import text


def detect_market_phase(engine) -> str:
    """根据链上活跃地址与 SOPR 判断市场阶段。

    返回值为 "bull"、"bear" 或 "range"。
    """
    q = text(
        "SELECT timestamp, metric, value "
        "FROM cm_onchain_metrics "
        "WHERE metric IN ('AdrActCnt','Sopr') "
        "ORDER BY timestamp DESC LIMIT 120"
    )
    df = pd.read_sql(q, engine, parse_dates=["timestamp"])
    if df.empty:
        return "range"

    df = (
        df.pivot(index="timestamp", columns="metric", values="value")
        .reset_index()
        .sort_values("timestamp")
    )
    if df.empty:
        return "range"

    aa = df["AdrActCnt"].astype(float)
    sopr = df["Sopr"].astype(float)
    aa_ma = aa.rolling(window=30, min_periods=1).mean().iloc[-1]
    sopr_ma = sopr.rolling(window=30, min_periods=1).mean().iloc[-1]
    cur_aa = aa.iloc[-1]
    cur_sopr = sopr.iloc[-1]

    if cur_aa > aa_ma and cur_sopr > 1:
        return "bull"
    if cur_aa < aa_ma and cur_sopr < 1:
        return "bear"
    return "range"

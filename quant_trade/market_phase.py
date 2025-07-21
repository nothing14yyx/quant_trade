import pandas as pd
from sqlalchemy import text


def detect_market_phase(engine) -> str:
    """根据活跃地址与市值判断市场阶段。

    返回值为 "bull"、"bear" 或 "range"。
    """
    q = text(
        "SELECT timestamp, metric, value FROM cm_onchain_metrics "
        "WHERE metric IN ('AdrActCnt','CapMrktCurUSD','SplyAdrBal1Cnt','FeesUSD') "
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
    bal = pd.to_numeric(df.get("SplyAdrBal1Cnt"), errors="coerce")
    fees = pd.to_numeric(df.get("FeesUSD"), errors="coerce")

    aa_ma = aa.rolling(window=30, min_periods=1).mean().iloc[-1]
    cap_ma = cap.rolling(window=30, min_periods=1).mean().iloc[-1]
    bal_ma = bal.rolling(window=30, min_periods=1).mean().iloc[-1] if not bal.empty else None
    fees_ma = fees.rolling(window=30, min_periods=1).mean().iloc[-1] if not fees.empty else None

    cur_aa = aa.iloc[-1]
    cur_cap = cap.iloc[-1]
    cur_bal = bal.iloc[-1] if not bal.empty else None
    cur_fees = fees.iloc[-1] if not fees.empty else None

    bull_cond = cur_aa > aa_ma and cur_cap > cap_ma
    bear_cond = cur_aa < aa_ma and cur_cap < cap_ma
    if cur_bal is not None and bal_ma is not None:
        bull_cond &= cur_bal > bal_ma
        bear_cond &= cur_bal < bal_ma
    if cur_fees is not None and fees_ma is not None:
        bull_cond &= cur_fees > fees_ma
        bear_cond &= cur_fees < fees_ma

    if bull_cond:
        return "bull"
    if bear_cond:
        return "bear"
    return "range"

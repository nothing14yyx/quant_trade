import pandas as pd
from sqlalchemy import text
from pathlib import Path

from .config_manager import ConfigManager

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


def detect_market_phase(engine, config_path: str | Path = CONFIG_PATH):
    """根据活跃地址与市值判断市场阶段并计算相对强度 ``S``。

    默认读取 ``market_phase.symbol``（或 ``symbols`` 列表）来筛选交易对，
    当仅指定一个交易对时返回其阶段名称；
    多个交易对则返回 ``{symbol: {"phase": str, "S": float}}`` 的字典。"""
    cfg = ConfigManager(config_path).get("market_phase", {})
    metrics = cfg.get("metrics", ["AdrActCnt", "CapMrktCurUSD", "FeeTotUSD"])
    symbols_cfg = cfg.get("symbols")
    if symbols_cfg is None:
        symbol = cfg.get("symbol", "BTCUSDT")
        symbols = [symbol]
    else:
        symbols = symbols_cfg if isinstance(symbols_cfg, list) else [symbols_cfg]

    base = ["AdrActCnt", "CapMrktCurUSD"]
    metrics = list(dict.fromkeys(base + [m for m in metrics if m not in base]))
    placeholders = ",".join(f"'{m}'" for m in metrics)
    q = text(
        (
            "SELECT timestamp, metric, value FROM cm_onchain_metrics "
            "WHERE symbol=:symbol AND metric IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT 120"
        ).format(placeholders=placeholders)
    )

    results = {}
    for symbol in symbols:
        df = pd.read_sql(q, engine, params={"symbol": symbol}, parse_dates=["timestamp"])
        if df.empty:
            results[symbol] = {"phase": "range", "S": 0.0}
            continue

        df = (
            df.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="mean")
            .sort_index()
        )

        ratios = []
        for m in metrics:
            series = pd.to_numeric(df.get(m), errors="coerce")
            if series.dropna().empty:
                continue
            ma = series.rolling(window=30, min_periods=1).mean().iloc[-1]
            cur = series.iloc[-1]
            ratio = cur / ma if ma else float("nan")
            ratios.append(ratio)

        if not ratios:
            results[symbol] = {"phase": "range", "S": 0.0}
            continue

        S = float(pd.Series(ratios).mean())
        if S > 1.05:
            phase = "bull"
        elif S < 0.95:
            phase = "bear"
        else:
            phase = "range"
        results[symbol] = {"phase": phase, "S": S}

    if len(symbols) == 1:
        return results[symbols[0]]["phase"]
    return results

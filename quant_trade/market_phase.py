import pandas as pd
from sqlalchemy import text
from pathlib import Path

from .config_manager import ConfigManager

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    z = (series - mean) / std
    return z.replace([float("inf"), float("-inf")], pd.NA)


def _normalize_weights(weights: dict, metrics: list[str]) -> pd.Series:
    if not weights:
        return pd.Series(1.0, index=metrics) / len(metrics)
    ser = pd.Series({m: weights.get(m, 1.0) for m in metrics}, dtype=float)
    total = ser.sum()
    if total == 0:
        ser[:] = 1.0
        total = len(metrics)
    return ser / total


def _phase_from_score(score: float) -> str:
    if pd.isna(score) or score == 0:
        return "range"
    return "bull" if score > 0 else "bear"


def detect_market_phase(engine, config_path: str | Path = CONFIG_PATH) -> dict:
    """根据活跃地址与市值判断市场阶段。"""

    cfg = ConfigManager(config_path).get("market_phase", {})
    metrics = cfg.get("metrics", ["AdrActCnt", "CapMrktCurUSD", "FeeTotUSD"])
    window = cfg.get("window", 30)
    weights_cfg = cfg.get("weights", {})

    symbols_cfg = cfg.get("symbols")
    if symbols_cfg is None:
        symbol = cfg.get("symbol", "BTCUSDT")
        symbols = [symbol]
    else:
        symbols = symbols_cfg if isinstance(symbols_cfg, list) else [symbols_cfg]

    placeholders = ",".join(f"'{m}'" for m in metrics)
    q = text(
        (
            "SELECT timestamp, metric, value FROM cm_onchain_metrics "
            "WHERE symbol=:symbol AND metric IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT 120"
        ).format(placeholders=placeholders)
    )

    metric_weights = _normalize_weights(weights_cfg, metrics)
    results: dict[str, dict] = {}
    caps: dict[str, float] = {}
    latest_ts = None

    for sym in symbols:
        df = pd.read_sql(q, engine, params={"symbol": sym}, parse_dates=["timestamp"])
        if df.empty:
            continue
        pivot = (
            df.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="mean")
            .sort_index()
        )

        scores = []
        for m in metrics:
            series = pd.to_numeric(pivot.get(m), errors="coerce")
            if series.dropna().empty:
                scores.append(0.0)
            else:
                z = _rolling_zscore(series, window).iloc[-1]
                scores.append(0.0 if pd.isna(z) else float(z))

        s_chain = float((metric_weights * pd.Series(scores, index=metrics)).sum())
        results[sym] = {"S": s_chain, "phase": _phase_from_score(s_chain)}

        cap_series = pd.to_numeric(pivot.get("CapMrktCurUSD"), errors="coerce")
        if not cap_series.dropna().empty:
            caps[sym] = float(cap_series.iloc[-1])

        ts = pivot.index.max()
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts

    if not results:
        return {"TOTAL": {"phase": "range"}}

    if caps:
        total_cap = sum(caps.values())
        chain_weights = {s: caps.get(s, 0) / total_cap for s in results}
    else:
        chain_weights = {s: 1 / len(results) for s in results}

    s_total = sum(chain_weights[s] * results[s]["S"] for s in results)
    results["TOTAL"] = {"S": s_total, "phase": _phase_from_score(s_total)}
    if latest_ts is not None:
        results["latest_timestamp"] = latest_ts
    results["window"] = window
    return results

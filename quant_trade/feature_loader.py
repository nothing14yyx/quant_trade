"""Utility functions to load features and metrics from the database."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import numpy as np
from sqlalchemy import text

from quant_trade.utils.db import load_config, connect_mysql
from quant_trade.utils.helper import (
    calc_features_raw,
    calc_order_book_features,
)
from quant_trade.feature_engineering import calc_cross_features
from quant_trade.utils.robust_scaler import (
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)
from quant_trade.data_loader import compute_vix_proxy


logger = logging.getLogger(__name__)


def load_latest_klines(engine, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """从数据库加载指定周期的最新K线并返回DataFrame，并加入 btc/eth 收盘价"""
    query = (
        "SELECT open_time, open, high, low, close, volume, fg_index, funding_rate, "
        "taker_buy_base,taker_buy_quote, cg_price, cg_market_cap, cg_total_volume "
        f"FROM klines WHERE symbol='{symbol}' AND `interval`='{interval}' "
        f"ORDER BY open_time DESC LIMIT {limit}"
    )
    df = pd.read_sql(query, engine, parse_dates=["open_time"])
    df = df.sort_values("open_time")

    if symbol != "BTCUSDT":
        btc_q = (
            "SELECT open_time, close FROM klines "
            f"WHERE symbol='BTCUSDT' AND `interval`='{interval}' "
            f"ORDER BY open_time DESC LIMIT {limit}"
        )
        btc_df = pd.read_sql(btc_q, engine, parse_dates=["open_time"])
        if not btc_df.empty:
            btc_df = btc_df.rename(columns={"close": "btc_close"}).sort_values("open_time")
            df = pd.merge_asof(df, btc_df, on="open_time", direction="backward")

    if symbol != "ETHUSDT":
        eth_q = (
            "SELECT open_time, close FROM klines "
            f"WHERE symbol='ETHUSDT' AND `interval`='{interval}' "
            f"ORDER BY open_time DESC LIMIT {limit}"
        )
        eth_df = pd.read_sql(eth_q, engine, parse_dates=["open_time"])
        if not eth_df.empty:
            eth_df = eth_df.rename(columns={"close": "eth_close"}).sort_values("open_time")
            df = pd.merge_asof(df, eth_df, on="open_time", direction="backward")

    return df


def prepare_features(
    df: pd.DataFrame,
    period: str,
    params: dict,
    symbol: str,
    offset: int = 0,
) -> tuple[dict, dict]:
    """计算单周期特征并返回(缩放后的dict, 原始dict)"""
    feats = calc_features_raw(
        df.set_index("open_time"),
        period,
        symbol=symbol,
    )
    if len(feats) <= offset:
        raise IndexError("insufficient data for requested offset")
    raw = feats.iloc[-1 - offset]

    if "funding_rate" in feats.columns:
        feats[f"funding_rate_{period}"] = feats["funding_rate"]

    last = feats.iloc[[-1 - offset]].copy()
    last["symbol"] = symbol
    last = apply_robust_z_with_params(last, params)
    scaled = last.drop(columns=["symbol"]).iloc[0]

    scaled_dict = scaled.to_dict()
    raw_dict = raw.to_dict()
    if "funding_rate" in raw_dict:
        raw_dict[f"funding_rate_{period}"] = raw_dict["funding_rate"]
        scaled_dict[f"funding_rate_{period}"] = scaled_dict.get("funding_rate")

    return scaled_dict, raw_dict


def prepare_all_features(
    engine,
    symbol: str,
    params: dict,
    offset: int = 0,
) -> tuple[dict, dict, dict, dict, dict, dict] | None:
    """加载多周期K线并构造包含跨周期特征的字典"""
    df1h = load_latest_klines(engine, symbol, "1h")
    df4h = load_latest_klines(engine, symbol, "4h")
    dfd1 = load_latest_klines(engine, symbol, "d1")
    df5m = load_latest_klines(engine, symbol, "5m")
    df15m = load_latest_klines(engine, symbol, "15m")

    f1h_df = calc_features_raw(
        df1h.set_index("open_time"),
        "1h",
        symbol=symbol,
    )
    f4h_df = calc_features_raw(
        df4h.set_index("open_time"),
        "4h",
        symbol=symbol,
    )
    fd1_df = calc_features_raw(
        dfd1.set_index("open_time"),
        "d1",
        symbol=symbol,
    )
    if f1h_df is None or f4h_df is None or fd1_df is None:
        return None

    if (
        "funding_rate_anom_d1" not in fd1_df
        or fd1_df["funding_rate_anom_d1"].isna().all()
    ) and "funding_rate" in df1h:
        fr = pd.to_numeric(df1h["funding_rate"], errors="coerce")
        fr.index = pd.to_datetime(df1h["open_time"], errors="coerce")
        fr_d = fr.resample("1D").mean()
        fr_ema = fr_d.ewm(span=24, adjust=False).mean()
        fr_anom = (fr_d - fr_ema).reindex(fd1_df.index, method="ffill")
        fd1_df["funding_rate_anom_d1"] = fr_anom

    merged = calc_cross_features(f1h_df, f4h_df, fd1_df)
    hour = merged["open_time"].dt.hour.astype(float)
    merged["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24)
    merged["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = merged["open_time"].dt.dayofweek.astype(float)
    merged["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    merged["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    if len(merged) <= offset:
        raise IndexError("insufficient cross period data for requested offset")
    cross_last = merged.iloc[[-1 - offset]].copy()
    cross_last["symbol"] = symbol
    cross_scaled = apply_robust_z_with_params(cross_last, params).drop(columns=["symbol"]).iloc[0]

    cross_cols = [
        c
        for c in cross_last.columns
        if any(
            x in c
            for x in [
                "_1h_4h",
                "_1h_d1",
                "_4h_d1",
                "hour_of_day_sin",
                "hour_of_day_cos",
                "day_of_week_sin",
                "day_of_week_cos",
            ]
        )
    ]

    scaled1h, raw1h = prepare_features(df1h, "1h", params, symbol, offset)
    scaled4h, raw4h = prepare_features(df4h, "4h", params, symbol, offset)
    scaledd1, rawd1 = prepare_features(dfd1, "d1", params, symbol, offset)

    if not df5m.empty:
        f5m = calc_features_raw(
            df5m.set_index("open_time"),
            "5m",
            symbol=symbol,
        )
        if f5m is not None and "pct_chg1_5m" in f5m:
            chg5 = f5m["pct_chg1_5m"].shift(1)
            f5m["mom_5m_roll1h"] = chg5.rolling(12, min_periods=1).mean()
            f5m["mom_5m_roll1h_std"] = chg5.rolling(12, min_periods=1).std()
            idx5 = -1 - offset * 12
            if abs(idx5) <= len(f5m):
                last5 = f5m.iloc[[idx5]].copy()
                last5["symbol"] = symbol
                s5 = apply_robust_z_with_params(last5, params).drop(columns=["symbol"]).iloc[0]
                for c in ["mom_5m_roll1h", "mom_5m_roll1h_std"]:
                    scaled1h[c] = s5.get(c)
                    raw1h[c] = last5[c].iloc[0]

    if not df15m.empty:
        f15m = calc_features_raw(
            df15m.set_index("open_time"),
            "15m",
            symbol=symbol,
        )
        if f15m is not None and "pct_chg1_15m" in f15m:
            chg15 = f15m["pct_chg1_15m"].shift(1)
            f15m["mom_15m_roll1h"] = chg15.rolling(4, min_periods=1).mean()
            f15m["mom_15m_roll1h_std"] = chg15.rolling(4, min_periods=1).std()
            idx15 = -1 - offset * 4
            if abs(idx15) <= len(f15m):
                last15 = f15m.iloc[[idx15]].copy()
                last15["symbol"] = symbol
                s15 = apply_robust_z_with_params(last15, params).drop(columns=["symbol"]).iloc[0]
                for c in [
                    "mom_15m_roll1h",
                    "mom_15m_roll1h_std",
                    "rsi_fast_15m",
                    "stoch_fast_15m",
                ]:
                    scaled1h[c] = s15.get(c)
                    raw1h[c] = last15[c].iloc[0]

    for col in cross_cols:
        if col not in scaled1h:
            scaled1h[col] = cross_scaled[col]
            raw1h[col] = cross_last[col].iloc[0]
        if col not in scaled4h:
            scaled4h[col] = cross_scaled[col]
            raw4h[col] = cross_last[col].iloc[0]
        if col not in scaledd1:
            scaledd1[col] = cross_scaled[col]
            rawd1[col] = cross_last[col].iloc[0]

    common_1h_cols = [
        "pct_chg1_1h",
        "hv_30d_1h",
        "obv_delta_1h",
        "upper_wick_ratio_1h",
        "kurtosis_1h",
        "bear_streak_1h",
        "cci_delta_1h",
    ]
    for col in common_1h_cols:
        if col in scaled1h:
            scaled4h[col] = scaled1h[col]
            scaledd1[col] = scaled1h[col]
            raw4h[col] = raw1h.get(col)
            rawd1[col] = raw1h.get(col)

    if "upper_wick_ratio_4h" in scaled4h:
        scaledd1["upper_wick_ratio_4h"] = scaled4h["upper_wick_ratio_4h"]
        rawd1["upper_wick_ratio_4h"] = raw4h.get("upper_wick_ratio_4h")

    if "pct_chg6_4h" in scaled4h:
        scaledd1["pct_chg6_4h"] = scaled4h["pct_chg6_4h"]
        rawd1["pct_chg6_4h"] = raw4h.get("pct_chg6_4h")

    if "funding_rate_anom_d1" in scaledd1:
        scaled1h["funding_rate_anom_d1"] = scaledd1["funding_rate_anom_d1"]
        scaled4h["funding_rate_anom_d1"] = scaledd1["funding_rate_anom_d1"]
        raw1h["funding_rate_anom_d1"] = rawd1.get("funding_rate_anom_d1")
        raw4h["funding_rate_anom_d1"] = rawd1.get("funding_rate_anom_d1")

    return scaled1h, scaled4h, scaledd1, raw1h, raw4h, rawd1


def load_latest_open_interest(engine, symbol: str) -> dict | None:
    """获取最新两条持仓量数据并计算变化率和 vix_proxy"""
    q = text(
        "SELECT timestamp, open_interest FROM open_interest "
        "WHERE symbol=:s ORDER BY timestamp DESC LIMIT 2"
    )
    df = pd.read_sql(q, engine, params={"s": symbol}, parse_dates=["timestamp"])
    if df.empty:
        return None
    latest = df.iloc[0]
    if len(df) > 1 and df.iloc[1]["open_interest"]:
        prev = df.iloc[1]
        prev_val = prev["open_interest"]
        oi_chg = (latest["open_interest"] - prev_val) / prev_val if prev_val else None
    else:
        oi_chg = None

    fr_q = text(
        "SELECT fundingRate FROM funding_rate "
        "WHERE symbol=:s ORDER BY fundingTime DESC LIMIT 1"
    )
    fr_df = pd.read_sql(fr_q, engine, params={"s": symbol})
    funding_rate = float(fr_df["fundingRate"].iloc[0]) if not fr_df.empty else None
    vix_p = compute_vix_proxy(funding_rate, oi_chg)
    return {
        "timestamp": latest["timestamp"],
        "open_interest": float(latest["open_interest"]),
        "oi_chg": float(oi_chg) if oi_chg is not None else None,
        "vix_proxy": float(vix_p) if vix_p is not None else None,
    }


def load_order_book_imbalance(engine, symbol: str) -> float | None:
    """读取最新的 order_book 快照计算口失衡值"""
    q = text(
        "SELECT timestamp, bids, asks FROM order_book "
        "WHERE symbol=:s ORDER BY timestamp DESC LIMIT 1"
    )
    df = pd.read_sql(q, engine, params={"s": symbol})
    if df.empty:
        return None
    feats = calc_order_book_features(df)
    val = feats["bid_ask_imbalance"].iloc[0]
    return float(val) if pd.notnull(val) else None


def load_hot_sector(engine) -> dict | None:
    """获取当前热门板块及强度"""
    q = (
        "SELECT id, name, market_cap, market_cap_change_24h, volume_24h, top_3_coins "
        "FROM cg_category_stats "
        "WHERE updated_at = (SELECT MAX(updated_at) FROM cg_category_stats)"
    )
    df = pd.read_sql(q, engine)
    df = df.dropna(subset=["name", "volume_24h"])
    if df.empty:
        return None
    df["volume_24h"] = pd.to_numeric(df["volume_24h"], errors="coerce")
    df = df.dropna(subset=["volume_24h"])
    if df.empty:
        return None
    total = df["volume_24h"].sum()
    df = df.sort_values("volume_24h", ascending=False)
    top = df.iloc[0]
    strength = float(top["volume_24h"]) / total if total else None
    return {"hot_sector": top["name"], "hot_sector_strength": strength}


def load_global_metrics(engine, symbol: str | None = None) -> dict | None:
    """返回最新的 CoinGecko 全局指标及变化率，并可按币种给出板块相关性"""
    q = (
        "SELECT timestamp, total_market_cap, total_volume, btc_dominance, eth_dominance "
        "FROM cg_global_metrics ORDER BY timestamp DESC LIMIT 2"
    )
    df = pd.read_sql(q, engine, parse_dates=["timestamp"])
    if df.empty:
        return None
    latest = df.iloc[0]
    if len(df) > 1:
        prev = df.iloc[1]
        pct = lambda cur, prev_val: (cur - prev_val) / prev_val if prev_val else None
        btc_dom_chg = pct(latest["btc_dominance"], prev["btc_dominance"])
        mcap_growth = pct(latest["total_market_cap"], prev["total_market_cap"])
        vol_chg = pct(latest["total_volume"], prev["total_volume"])
        eth_dom_chg = pct(latest["eth_dominance"], prev["eth_dominance"])
        btc_mcap_prev = prev["total_market_cap"] * prev["btc_dominance"] / 100
        btc_mcap_cur = latest["total_market_cap"] * latest["btc_dominance"] / 100
        alt_prev = prev["total_market_cap"] - btc_mcap_prev
        alt_cur = latest["total_market_cap"] - btc_mcap_cur
        btc_mcap_growth = pct(btc_mcap_cur, btc_mcap_prev)
        alt_mcap_growth = pct(alt_cur, alt_prev)
    else:
        btc_dom_chg = mcap_growth = vol_chg = None
        eth_dom_chg = btc_mcap_growth = alt_mcap_growth = None

    metrics = {
        "timestamp": latest["timestamp"],
        "btc_dom_chg": float(btc_dom_chg) if btc_dom_chg is not None else None,
        "mcap_growth": float(mcap_growth) if mcap_growth is not None else None,
        "vol_chg": float(vol_chg) if vol_chg is not None else None,
        "btc_dominance": float(latest["btc_dominance"]),
        "total_market_cap": float(latest["total_market_cap"]),
        "total_volume": float(latest["total_volume"]),
        "eth_dominance": float(latest["eth_dominance"]),
        "eth_dom_chg": float(eth_dom_chg) if eth_dom_chg is not None else None,
        "btc_mcap_growth": float(btc_mcap_growth) if btc_mcap_growth is not None else None,
        "alt_mcap_growth": float(alt_mcap_growth) if alt_mcap_growth is not None else None,
    }
    oi = load_latest_open_interest(engine, "BTCUSDT")
    if oi and oi.get("vix_proxy") is not None:
        metrics["vix_proxy"] = oi["vix_proxy"]
    hot = load_hot_sector(engine)
    if hot:
        metrics.update(hot)
        if symbol is not None:
            q = text(
                "SELECT categories FROM cg_coin_categories WHERE symbol=:s"
            )
            df_cat = pd.read_sql(q, engine, params={"s": symbol})
            if not df_cat.empty:
                cats = df_cat["categories"].iloc[0]
                if isinstance(cats, str) and cats:
                    arr = [c.strip() for c in cats.split(",") if c.strip()]
                    metrics["sector_corr"] = 1.0 if hot["hot_sector"] in arr else 0.0
    return metrics


def load_symbol_categories(engine) -> dict:
    """读取币种与所属板块的映射"""
    df = pd.read_sql("SELECT symbol, categories FROM cg_coin_categories", engine)
    return {r["symbol"].upper(): r["categories"] for _, r in df.iterrows()}


# ─── Async wrappers ───────────────────────────────────────────────────────────

async def prepare_all_features_async(engine, symbol: str, params: dict, offset: int = 0) -> Any:
    return await asyncio.to_thread(prepare_all_features, engine, symbol, params, offset)


async def load_global_metrics_async(engine, symbol: str | None = None) -> Any:
    return await asyncio.to_thread(load_global_metrics, engine, symbol)


async def load_latest_open_interest_async(engine, symbol: str) -> Any:
    return await asyncio.to_thread(load_latest_open_interest, engine, symbol)


async def load_order_book_imbalance_async(engine, symbol: str) -> Any:
    return await asyncio.to_thread(load_order_book_imbalance, engine, symbol)


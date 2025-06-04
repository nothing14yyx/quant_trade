# utils/helper.py  — 去掉外部指标后的版本

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ta

# --------- 全局配置（只保留 fg_index 和 funding_rate，如果这两个也不需要可以一并删掉） ---------
LAGGED_COLS = [
    "fg_index",
    "funding_rate",
]


# ------------------------------------------------------------------ #
# ❶  训练阶段: 统计 robust-z 参数并保存
# ------------------------------------------------------------------ #
def fit_scaler(df: pd.DataFrame, save_path: str | Path = "scaler.json") -> Dict[str, Tuple[float, float, float, float]]:
    """
    统计每个数值列的 1%/99% 分位，裁剪后计算均值和标准差，保存为 robust-z 参数。
    """
    params: Dict[str, Tuple[float, float, float, float]] = {}
    for col in df.columns:
        arr = df[col].astype(float).values
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, p1, p99)
        mu, std = arr.mean(), arr.std() if arr.std() > 1e-8 else 1.0
        params[col] = (float(p1), float(p99), float(mu), float(std))

    Path(save_path).write_text(json.dumps(params, indent=2))
    return params


# ------------------------------------------------------------------ #
# ❷  推理 / 通用: 计算全部特征（去掉外部指标后的版本）
# ------------------------------------------------------------------ #
def calc_features_full(df_raw: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    统一特征计算入口，去掉所有 vix/dxy/链上等外部指标。
    只保留：基础价格、成交量、FG 指数、funding_rate，以及纯技术指标的计算。
    """
    df = df_raw.copy()

    # ---------- 滞后列一次性 forward-fill ----------
    # 只对 fg_index 和 funding_rate 做前向填充
    df[LAGGED_COLS] = df[LAGGED_COLS].ffill()

    # ---------- 价格列 ----------
    open_, high, low, close, vol = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feats = pd.DataFrame(index=df.index)

    # —— 1. 趋势 & 动量 ——
    feats[f"ema10_ema50_{period}"] = ta.trend.ema_indicator(close, 10) - ta.trend.ema_indicator(close, 50)
    feats[f"rsi_{period}"]         = ta.momentum.rsi(close, 14)
    feats[f"rsi_slope_{period}"]   = feats[f"rsi_{period}"].diff()

    # —— 2. 波动率 ——
    def safe_atr(_high, _low, _close, win: int = 14):
        if len(_high) < win:
            return pd.Series(np.nan, index=_high.index)
        return ta.volatility.average_true_range(_high, _low, _close, win)

    feats[f"atr_pct_{period}"] = safe_atr(high, low, close) / close

    # Bollinger & Keltner
    try:
        bb = ta.volatility.BollingerBands(close, 20, 2)
        feats[f"bb_width_{period}"] = bb.bollinger_hband() - bb.bollinger_lband()
    except Exception:
        feats[f"bb_width_{period}"] = np.nan

    try:
        kc = ta.volatility.KeltnerChannel(high, low, close, 20, 10)
        feats[f"kc_perc_{period}"] = (
            (close - kc.keltner_channel_lband()) /
            (kc.keltner_channel_hband() - kc.keltner_channel_lband())
        )
    except Exception:
        feats[f"kc_perc_{period}"] = np.nan

    # —— 3. 量能 ——
    feats[f"vol_roc_{period}"]      = vol.pct_change()
    feats[f"vol_ma_ratio_{period}"] = vol / vol.rolling(20).mean()

    # —— 4. 连涨 / 连跌 (避免未来泄漏) ——
    bull = (close > open_).astype(int)
    bear = (close < open_).astype(int)
    feats[f"bull_streak_{period}"] = bull.rolling(3, closed="left").sum()
    feats[f"bear_streak_{period}"] = bear.rolling(3, closed="left").sum()

    # —— 5. 滞后列（仅 fg_index 与 funding_rate）+ Δ ——
    for col in LAGGED_COLS:
        feats[f"{col}_{period}"] = df[col]
        vals = pd.to_numeric(df[col], errors="coerce")
        feats[f"{col}_delta_{period}"] = vals.diff()

    # —— 6. 简单交互 —— 存在且非全空才计算
    pairs = [("rsi", "vol_ma_ratio"), ("atr_pct", "bb_width")]
    for a, b in pairs:
        ca, cb = f"{a}_{period}", f"{b}_{period}"
        if ca in feats and cb in feats and feats[ca].notna().any() and feats[cb].notna().any():
            feats[f"{a}_mul_{b}_{period}"] = feats[ca] * feats[cb]

    # —— 7. Robust-z 统一缩放 ——
    for col in feats.columns:
        # 跳过标记列
        if feats[col].dtype == "int64":
            continue
        arr = feats[col].astype(float).values
        mask = np.isfinite(arr)
        if not mask.any():
            continue
        p1, p99 = np.percentile(arr[mask], [1, 99])
        arr = np.clip(arr, p1, p99)
        mu, std = arr[mask].mean(), arr[mask].std() if arr[mask].std() > 1e-8 else 1.0
        feats[col] = (arr - mu) / std

    return feats

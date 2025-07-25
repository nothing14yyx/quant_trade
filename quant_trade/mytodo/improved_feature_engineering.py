"""
改进版特征工程示例
==================

本模块示例展示了如何重构 `calc_cross_features` 功能，以通过循环动态生成跨周期特征，
从而简化代码和提升可维护性。它不会包含原项目的所有逻辑，仅用作参考。使用时，可将
下述函数合并至原项目 `feature_engineering.py` 中，并在 `_calc_symbol_features`
方法中替换调用。

"""

from __future__ import annotations

import pandas as pd
import numpy as np


def calc_cross_features_v2(
    df1h: pd.DataFrame, df4h: pd.DataFrame, df1d: pd.DataFrame
) -> pd.DataFrame:
    """
    改进版跨周期特征生成函数。

    参数
    ------
    df1h, df4h, df1d : pandas.DataFrame
        分别为 1 小时、4 小时、日线周期的特征 DataFrame，须包含 ``open_time``、
        ``close``、``sma_5``、``sma_10``、``sma_20``、``atr_pct``、``bb_width``、
        ``rsi``、``macd_hist``、``vol_ma_ratio`` 等列。

    返回
    ------
    pandas.DataFrame
        包含跨周期特征的合并表，列名与原版相兼容，并自动扩展比值/差值。

    示例
    ------
    >>> merged = calc_cross_features_v2(f1h, f4h, f1d)
    >>> merged.columns
    Index([... 'ma_ratio_1h_4h', 'rsi_diff_1h_4h', 'macd_hist_4h_mul_bb_width_1h', ...])
    """
    # 复制输入，防止原数据被修改
    f1h, f4h, f1d = df1h.copy(), df4h.copy(), df1d.copy()

    # 格式化 open_time
    for df in (f1h, f4h, f1d):
        if "open_time" in df.columns:
            df.reset_index(drop=True, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")
        df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")

    # 处理社交情绪
    if "social_sentiment" in f1h.columns:
        ss = f1h["social_sentiment"].astype(float)
        f1h["social_sentiment_1h"] = ss
        f1h["social_sentiment_4h"] = ss.rolling(4, min_periods=1).mean()
        f1h = f1h.drop(columns=["social_sentiment"])

    # 重命名收盘价
    f1h = f1h.rename(columns={"close": "close_1h"})
    f4h = f4h.rename(columns={"close": "close_4h"})
    f1d = f1d.rename(columns={"close": "close_d1"})

    # 计算 close_time 用于 merge_asof
    f4h["close_time_4h"] = f4h["open_time"] + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
    f1d["close_time_d1"] = f1d["open_time"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    f4h = f4h.drop(columns=["open_time"])
    f1d = f1d.drop(columns=["open_time"])

    # 合并到 1h 基准表
    merged = pd.merge_asof(
        f1h.sort_values("open_time"),
        f4h.sort_values("close_time_4h"),
        left_on="open_time",
        right_on="close_time_4h",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("open_time"),
        f1d.sort_values("close_time_d1"),
        left_on="open_time",
        right_on="close_time_d1",
        direction="backward",
    )

    periods = ["1h", "4h", "d1"]
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            a, b = periods[i], periods[j]
            # 收盘价差
            merged[f"close_spread_{a}_{b}"] = merged[f"close_{a}"] - merged[f"close_{b}"]
            # MA (SMA10) 比值
            if f"sma_10_{a}" in merged.columns and f"sma_10_{b}" in merged.columns:
                merged[f"ma_ratio_{a}_{b}"] = merged[f"sma_10_{a}"] / merged[f"sma_10_{b}"].replace(0, np.nan)
            # ATR 比值
            if f"atr_pct_{a}" in merged.columns and f"atr_pct_{b}" in merged.columns:
                merged[f"atr_pct_ratio_{a}_{b}"] = merged[f"atr_pct_{a}"] / merged[f"atr_pct_{b}"].replace(0, np.nan)
            # 布林带宽度比值
            if f"bb_width_{a}" in merged.columns and f"bb_width_{b}" in merged.columns:
                merged[f"bb_width_ratio_{a}_{b}"] = merged[f"bb_width_{a}"] / merged[f"bb_width_{b}"].replace(0, np.nan)
            # RSI 差值
            if f"rsi_{a}" in merged.columns and f"rsi_{b}" in merged.columns:
                merged[f"rsi_diff_{a}_{b}"] = merged[f"rsi_{a}"] - merged[f"rsi_{b}"]
            # MACD 直方图差值
            if f"macd_hist_{a}" in merged.columns and f"macd_hist_{b}" in merged.columns:
                merged[f"macd_hist_diff_{a}_{b}"] = merged[f"macd_hist_{a}"] - merged[f"macd_hist_{b}"]
            # 成交量均值比值
            if f"vol_ma_ratio_{a}" in merged.columns and f"vol_ma_ratio_{b}" in merged.columns:
                merged[f"vol_ratio_{a}_{b}"] = merged[f"vol_ma_ratio_{a}"] / merged[f"vol_ma_ratio_{b}"].replace(0, np.nan)

    # 交叉乘积特征
    if {"macd_hist_4h", "bb_width_1h"}.issubset(merged.columns):
        merged["macd_hist_4h_mul_bb_width_1h"] = merged["macd_hist_4h"] * merged["bb_width_1h"]
    if {"rsi_1h", "vol_ma_ratio_4h"}.issubset(merged.columns):
        merged["rsi_1h_mul_vol_ma_ratio_4h"] = merged["rsi_1h"] * merged["vol_ma_ratio_4h"]
    if {"macd_hist_1h", "bb_width_4h"}.issubset(merged.columns):
        merged["macd_hist_1h_mul_bb_width_4h"] = merged["macd_hist_1h"] * merged["bb_width_4h"]

    # SMA5/20 比值
    if {"sma_5_1h", "sma_20_1h"}.issubset(merged.columns):
        merged["ma_ratio_5_20"] = merged["sma_5_1h"] / merged["sma_20_1h"].replace(0, np.nan)

    return merged
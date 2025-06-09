import numpy as np

# Pandas TA expects numpy.NaN constant which was removed in newer numpy versions
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta

def assign_safe(feats: pd.DataFrame, name: str, series):
    feats[name] = np.asarray(series, dtype="float64")
    # print(f"{name}: {feats[name].dtype}")

def calc_mfi_np(high, low, close, volume, window=14):
    # 1. 典型价格
    tp = (high + low + close) / 3
    # 2. 原始 Money Flow
    mf = tp * volume
    # 3. 区分正/负流入
    pmf = np.where(tp > np.roll(tp, 1), mf, 0)
    nmf = np.where(tp < np.roll(tp, 1), mf, 0)
    # 第一行没有前一行，补0
    pmf[0] = 0
    nmf[0] = 0
    # 4. 滚动窗口求和
    sum_pmf = pd.Series(pmf).rolling(window).sum().to_numpy()
    sum_nmf = pd.Series(nmf).rolling(window).sum().to_numpy()
    # 5. MFI公式
    mfi = 100 * sum_pmf / (sum_pmf + sum_nmf)
    return mfi

def calc_features_raw(df: pd.DataFrame, period: str) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close", "volume"]:
        feats[col] = np.full(len(feats), np.nan, dtype="float64")

    for col in ["open", "high", "low", "close", "volume"]:
        assign_safe(feats, col, df[col].astype(float))

    if "fg_index" in df:
        assign_safe(feats, "fg_index", df["fg_index"].astype(float).ffill())
    if "funding_rate" in df:
        assign_safe(feats, "funding_rate", df["funding_rate"].astype(float).ffill())
        fr_ema = ta.ema(feats["funding_rate"], length=24)
        assign_safe(feats, f"funding_rate_anom_{period}", (feats["funding_rate"] - fr_ema))

    assign_safe(feats, f"ema_diff_{period}", ta.ema(feats["close"], 10) - ta.ema(feats["close"], 50))
    assign_safe(feats, f"sma_10_{period}", ta.sma(feats["close"], length=10))
    feats[f"pct_chg1_{period}"] = feats["close"].pct_change()
    feats[f"pct_chg3_{period}"] = feats["close"].pct_change(3)
    feats[f"pct_chg6_{period}"] = feats["close"].pct_change(6)
    assign_safe(feats, f"rsi_{period}", ta.rsi(feats["close"], length=14))
    feats[f"rsi_slope_{period}"] = feats[f"rsi_{period}"].diff()
    assign_safe(feats, f"atr_pct_{period}", ta.atr(feats["high"], feats["low"], feats["close"], length=14) / feats["close"])
    feats[f"atr_chg_{period}"] = feats[f"atr_pct_{period}"].diff()

    assign_safe(feats, f"adx_{period}", ta.adx(feats["high"], feats["low"], feats["close"], length=14)["ADX_14"])
    feats[f"adx_delta_{period}"] = feats[f"adx_{period}"].diff()
    assign_safe(feats, f"cci_{period}", ta.cci(feats["high"], feats["low"], feats["close"], length=14))
    feats[f"cci_delta_{period}"] = feats[f"cci_{period}"].diff()
    assign_safe(feats, f"mfi_{period}",
                calc_mfi_np(feats["high"].values, feats["low"].values, feats["close"].values, feats["volume"].values,
                            window=14))

    bb = ta.bbands(feats["close"], length=20)
    assign_safe(feats, f"bb_width_{period}", bb["BBU_20_2.0"] - bb["BBL_20_2.0"])
    feats[f"bb_width_chg_{period}"] = feats[f"bb_width_{period}"].diff()
    assign_safe(feats, f"boll_perc_{period}", (feats["close"] - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]).replace(0, np.nan))

    kc = ta.kc(feats["high"], feats["low"], feats["close"], length=20)
    assign_safe(feats, f"kc_perc_{period}", (feats["close"] - kc["KCLe_20_2"]) / (kc["KCUe_20_2"] - kc["KCLe_20_2"]).replace(0, np.nan))

    dc = ta.donchian(feats["high"], feats["low"], lower_length=20, upper_length=20)
    assign_safe(feats, f"donchian_perc_{period}", (feats["close"] - dc["DCL_20_20"]) / (dc["DCU_20_20"] - dc["DCL_20_20"]).replace(0, np.nan))
    assign_safe(feats, f"donchian_delta_{period}", dc["DCU_20_20"] - dc["DCL_20_20"])

    assign_safe(feats, f"vol_roc_{period}", ta.roc(feats["volume"], length=5))
    assign_safe(feats, f"vol_ma_ratio_{period}", feats["volume"] / ta.sma(feats["volume"], length=10).replace(0, np.nan))
    assign_safe(feats, f"vol_ma_ratio_long_{period}", feats["volume"] / ta.sma(feats["volume"], length=30).replace(0, np.nan))

    range_ = (feats["high"] - feats["low"]).replace(0, np.nan)
    body = (feats["close"] - feats["open"]).abs()
    assign_safe(feats, f"upper_wick_ratio_{period}", (feats["high"] - np.maximum(feats["open"], feats["close"])) / range_)
    assign_safe(feats, f"lower_wick_ratio_{period}", (np.minimum(feats["open"], feats["close"]) - feats["low"]) / range_)
    assign_safe(feats, f"body_ratio_{period}", body / range_)

    # === 新增：长影线与低波动突破等结构特征 ===
    upper_long = (feats[f"upper_wick_ratio_{period}"] > 0.6) & (feats[f"body_ratio_{period}"] < 0.3)
    lower_long = (feats[f"lower_wick_ratio_{period}"] > 0.6) & (feats[f"body_ratio_{period}"] < 0.3)
    assign_safe(feats, f"long_upper_shadow_{period}", upper_long.astype(float))
    assign_safe(feats, f"long_lower_shadow_{period}", lower_long.astype(float))

    sma_bbw = ta.sma(feats[f"bb_width_{period}"], length=20)
    vol_breakout = (feats[f"bb_width_{period}"] > sma_bbw * 1.5) & (feats[f"vol_ma_ratio_{period}"] > 1.5)
    assign_safe(feats, f"vol_breakout_{period}", vol_breakout.astype(float))

    feats[f"bull_streak_{period}"] = (
        feats["close"].gt(feats["open"]).astype(float)
        .groupby(feats["close"].le(feats["open"]).astype(int).cumsum())
        .cumsum()
    )
    feats[f"bear_streak_{period}"] = (
        feats["close"].lt(feats["open"]).astype(float)
        .groupby(feats["close"].ge(feats["open"]).astype(int).cumsum())
        .cumsum()
    )

    assign_safe(feats, f"rsi_mul_vol_ma_ratio_{period}", feats[f"rsi_{period}"] * feats[f"vol_ma_ratio_{period}"])
    assign_safe(feats, f"willr_{period}", ta.willr(feats["high"], feats["low"], feats["close"], length=14))

    macd = ta.macd(feats["close"])
    assign_safe(feats, f"macd_{period}", macd["MACD_12_26_9"])
    assign_safe(feats, f"macd_signal_{period}", macd["MACDs_12_26_9"])
    assign_safe(feats, f"macd_hist_{period}", macd["MACDh_12_26_9"])

    assign_safe(feats, f"obv_{period}", ta.obv(feats["close"], feats["volume"]))
    feats[f"obv_delta_{period}"] = feats[f"obv_{period}"].diff()

    st = ta.supertrend(feats["high"], feats["low"], feats["close"])
    assign_safe(feats, f"supertrend_dir_{period}", st.get("SUPERTd_7_3.0", pd.Series(index=df.index, data=np.nan)))

    # ======== CoinGecko 衍生特征 ========
    if "cg_price" in df:
        cg_price = pd.to_numeric(df["cg_price"], errors="coerce")
        assign_safe(feats, f"price_diff_cg_{period}", feats["close"] - cg_price)
        assign_safe(
            feats,
            f"price_ratio_cg_{period}",
            feats["close"] / cg_price.replace(0, np.nan),
        )

    if "cg_market_cap" in df:
        cg_mc = pd.to_numeric(df["cg_market_cap"], errors="coerce")
        assign_safe(feats, f"cg_market_cap_roc_{period}", cg_mc.pct_change())

    if "cg_total_volume" in df:
        cg_tv = pd.to_numeric(df["cg_total_volume"], errors="coerce")
        assign_safe(feats, f"cg_total_volume_roc_{period}", cg_tv.pct_change())
        assign_safe(
            feats,
            f"volume_cg_ratio_{period}",
            feats["volume"] / cg_tv.replace(0, np.nan),
        )

    return feats

def calc_features_full(df: pd.DataFrame, period: str) -> pd.DataFrame:
    feats = calc_features_raw(df, period)

    for col in feats.columns:
        if pd.api.types.is_numeric_dtype(feats[col]):
            arr = feats[col].astype(float).values
            p1, p99 = np.nanpercentile(arr, [1, 99])
            clipped = np.clip(arr, p1, p99)
            mu = np.nanmean(clipped)
            sigma = np.nanstd(clipped) + 1e-6
            feats[col] = ((clipped - mu) / sigma).astype("float64")

    flag_df = pd.DataFrame(index=feats.index)
    for col in feats.columns:
        flag_df[f"{col}_isnan"] = feats[col].isna().astype(int)

    feats = pd.concat([feats, flag_df], axis=1)
    feats = feats.fillna(0)
    return feats

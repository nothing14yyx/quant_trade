import numpy as np

import pandas as pd
import pandas_ta as ta
import json


def collect_feature_cols(cfg: dict, period: str) -> list[str]:
    """\
    根据 config 中的 ``feature_cols`` 字段返回指定周期的特征列表。

    兼容 ``{"1h": [..]}``、旧版 ``{"1h": {"up": [...]}}`` 与
    新版 ``{"1h": {"cls": [...], "vol": [...]}}`` 等格式，
    若为字典会合并各标签的特征并去重后返回。
    """
    cols = cfg.get("feature_cols", {}).get(period, [])
    if isinstance(cols, dict):
        union = set()
        for v in cols.values():
            if isinstance(v, list):
                union.update(v)
        cols = sorted(union)
    return cols


def _safe_ta(func, *args, index=None, cols=None, **kwargs):
    """调用 pandas_ta 指标函数, 在数据不足时返回指定 dtype 的 DataFrame。"""
    try:
        res = func(*args, **kwargs)
    except Exception:
        res = None

    if res is None:
        res = pd.DataFrame(index=index, columns=cols or ["val"], dtype="float64")
    else:
        if isinstance(res, pd.Series):
            res = res.to_frame()
        res = res.astype("float64")
        if cols is not None:
            for c in cols:
                if c not in res.columns:
                    res[c] = np.nan
            res = res[cols]
    return res


def assign_safe(feats: pd.DataFrame, name: str, series):
    """安全地向 feats 赋值，支持传入 Series 或单列 DataFrame。"""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("assign_safe 仅支持单列 DataFrame")
        series = series.iloc[:, 0]
    feats[name] = np.asarray(series, dtype="float64")
    # print(f"{name}: {feats[name].dtype}")


def calc_mfi_np(high, low, close, volume, window=14):
    """Return Money Flow Ratio and Money Flow Index"""
    tp = (high + low + close) / 3
    mf = tp * volume
    pmf = np.where(tp > np.roll(tp, 1), mf, 0)
    nmf = np.where(tp < np.roll(tp, 1), mf, 0)
    pmf[0] = 0
    nmf[0] = 0
    sum_pmf = pd.Series(pmf).rolling(window).sum().to_numpy()
    sum_nmf = pd.Series(nmf).rolling(window).sum().to_numpy()
    ratio = sum_pmf / (sum_nmf + 1e-12)
    mfi = np.divide(100 * sum_pmf, sum_pmf + sum_nmf + 1e-12)
    return ratio, mfi


def calc_features_raw(df: pd.DataFrame, period: str) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close", "volume"]:
        feats[col] = np.full(len(feats), np.nan, dtype="float64")

    price_cols = ["open", "high", "low", "close"]
    quantiles = df[price_cols].quantile([0.001, 0.999])
    lower = quantiles.loc[0.001]
    upper = quantiles.loc[0.999]
    df_clipped = df.copy()
    df_clipped[price_cols] = df_clipped[price_cols].clip(lower, upper, axis=1)

    for col in ["open", "high", "low", "close", "volume"]:
        assign_safe(feats, col, df_clipped[col].astype(float))

    if "fg_index" in df:
        fg = df["fg_index"].astype(float).ffill()
        assign_safe(feats, "fg_index", fg)
        # 额外提供按日粒度的情绪指标，便于多周期因子引用
        assign_safe(feats, "fg_index_d1", fg)
    if "funding_rate" in df:
        assign_safe(feats, "funding_rate", df["funding_rate"].astype(float).ffill())
        fr_ema = _safe_ta(ta.ema, feats["funding_rate"], length=24, index=df.index)
        fr_ema_s = fr_ema.iloc[:, 0]
        assign_safe(feats, f"funding_rate_anom_{period}", (feats["funding_rate"] - fr_ema_s))

    ema_short = _safe_ta(ta.ema, feats["close"], length=10, index=df.index)
    ema_short_s = ema_short.iloc[:, 0]
    if ema_short_s.isna().all():
        ema_short_s = feats["close"].ewm(span=10, adjust=False).mean()

    ema_long = _safe_ta(ta.ema, feats["close"], length=50, index=df.index)
    ema_long_s = ema_long.iloc[:, 0]
    if ema_long_s.isna().all():
        ema_long_s = feats["close"].ewm(span=50, adjust=False).mean()

    assign_safe(feats, f"ema_diff_{period}", ema_short_s - ema_long_s)
    assign_safe(feats, f"sma_5_{period}", _safe_ta(ta.sma, feats["close"], length=5, index=df.index))
    assign_safe(feats, f"sma_10_{period}", _safe_ta(ta.sma, feats["close"], length=10, index=df.index))
    assign_safe(feats, f"sma_20_{period}", _safe_ta(ta.sma, feats["close"], length=20, index=df.index))
    feats[f"pct_chg1_{period}"] = (
        feats["close"].pct_change(fill_method=None).fillna(0)
    )
    feats[f"pct_chg3_{period}"] = (
        feats["close"].pct_change(3, fill_method=None).fillna(0)
    )
    feats[f"pct_chg6_{period}"] = (
        feats["close"].pct_change(6, fill_method=None).fillna(0)
    )
    assign_safe(feats, f"rsi_{period}", _safe_ta(ta.rsi, feats["close"], length=14, index=df.index))
    feats[f"rsi_slope_{period}"] = feats[f"rsi_{period}"].diff()
    atr = _safe_ta(ta.atr, feats["high"], feats["low"], feats["close"], length=14, index=df.index)
    atr_s = atr.iloc[:, 0]
    assign_safe(feats, f"atr_pct_{period}", atr_s.div(feats["close"], axis=0))
    feats[f"atr_chg_{period}"] = feats[f"atr_pct_{period}"].diff()

    adx_df = _safe_ta(
        ta.adx,
        feats["high"],
        feats["low"],
        feats["close"],
        length=14,
        index=df.index,
        cols=["ADX_14", "DMP_14", "DMN_14"],
    )
    assign_safe(feats, f"adx_{period}", adx_df.get("ADX_14"))
    feats[f"adx_delta_{period}"] = feats[f"adx_{period}"].diff()
    assign_safe(
        feats, f"cci_{period}", _safe_ta(ta.cci, feats["high"], feats["low"], feats["close"], length=14, index=df.index)
    )
    feats[f"cci_delta_{period}"] = feats[f"cci_{period}"].diff().fillna(0)
    mfr, mfi = calc_mfi_np(
        feats["high"].values,
        feats["low"].values,
        feats["close"].values,
        feats["volume"].values,
        window=14,
    )
    assign_safe(feats, f"mfi_{period}", mfi)
    assign_safe(feats, f"money_flow_ratio_{period}", mfr)

    # VWAP 和随机指标
    assign_safe(
        feats,
        f"vwap_{period}",
        _safe_ta(
            ta.vwap,
            feats["high"],
            feats["low"],
            feats["close"],
            feats["volume"],
            index=df.index,
        ),
    )
    stoch = _safe_ta(
        ta.stoch,
        feats["high"],
        feats["low"],
        feats["close"],
        k=14,
        d=3,
        smooth_k=3,
        index=df.index,
        cols=["STOCHk_14_3_3", "STOCHd_14_3_3"],
    )
    stoch = stoch.reindex(df.index)
    assign_safe(feats, f"stoch_k_{period}", stoch.get("STOCHk_14_3_3"))
    assign_safe(feats, f"stoch_d_{period}", stoch.get("STOCHd_14_3_3"))

    bb = _safe_ta(
        ta.bbands,
        feats["close"],
        length=20,
        index=df.index,
        cols=["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"],
    )
    assign_safe(feats, f"bb_width_{period}", bb.get("BBU_20_2.0") - bb.get("BBL_20_2.0"))
    feats[f"bb_width_chg_{period}"] = feats[f"bb_width_{period}"].diff()
    assign_safe(
        feats,
        f"boll_perc_{period}",
        (feats["close"] - bb.get("BBL_20_2.0")) / (bb.get("BBU_20_2.0") - bb.get("BBL_20_2.0")).replace(0, np.nan),
    )

    kc = _safe_ta(
        ta.kc,
        feats["high"],
        feats["low"],
        feats["close"],
        length=20,
        index=df.index,
        cols=["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"],
    )
    assign_safe(
        feats,
        f"kc_perc_{period}",
        (feats["close"] - kc.get("KCLe_20_2")) / (kc.get("KCUe_20_2") - kc.get("KCLe_20_2")).replace(0, np.nan),
    )
    kc_width = kc.get("KCUe_20_2") - kc.get("KCLe_20_2")
    assign_safe(feats, f"kc_width_pct_chg_{period}", kc_width.pct_change(fill_method=None))

    ichi_raw = ta.ichimoku(feats["high"], feats["low"], feats["close"])
    if isinstance(ichi_raw, tuple):
        ichi_vis = ichi_raw[0]
    else:
        ichi_vis = ichi_raw
    if isinstance(ichi_vis, pd.DataFrame):
        assign_safe(feats, f"ichimoku_base_{period}", ichi_vis.get("IKS_26"))
        assign_safe(feats, f"ichimoku_conversion_{period}", ichi_vis.get("ITS_9"))
        assign_safe(
            feats,
            f"ichimoku_cloud_thickness_{period}",
            (ichi_vis.get("ISA_9") - ichi_vis.get("ISB_26")).abs(),
        )
    else:
        assign_safe(feats, f"ichimoku_base_{period}", pd.Series(index=df.index, dtype="float64"))
        assign_safe(feats, f"ichimoku_conversion_{period}", pd.Series(index=df.index, dtype="float64"))
        assign_safe(feats, f"ichimoku_cloud_thickness_{period}", pd.Series(index=df.index, dtype="float64"))

    dc = _safe_ta(
        ta.donchian,
        feats["high"],
        feats["low"],
        lower_length=20,
        upper_length=20,
        index=df.index,
        cols=["DCL_20_20", "DCM_20_20", "DCU_20_20"],
    )
    assign_safe(
        feats,
        f"donchian_perc_{period}",
        (feats["close"] - dc.get("DCL_20_20")) / (dc.get("DCU_20_20") - dc.get("DCL_20_20")).replace(0, np.nan),
    )
    assign_safe(feats, f"donchian_delta_{period}", dc.get("DCU_20_20") - dc.get("DCL_20_20"))

    assign_safe(feats, f"vol_roc_{period}", _safe_ta(ta.roc, feats["volume"], length=5, index=df.index))
    sma_vol_short = _safe_ta(ta.sma, feats["volume"], length=10, index=df.index).iloc[:, 0]
    assign_safe(
        feats,
        f"vol_ma_ratio_{period}",
        feats["volume"] / sma_vol_short.replace(0, np.nan),
    )
    sma_vol_long = _safe_ta(ta.sma, feats["volume"], length=30, index=df.index).iloc[:, 0]
    assign_safe(
        feats,
        f"vol_ma_ratio_long_{period}",
        feats["volume"] / sma_vol_long.replace(0, np.nan),
    )

    if "taker_buy_base" in df:
        buy_vol = pd.to_numeric(df["taker_buy_base"], errors="coerce")
        sell_vol = feats["volume"] - buy_vol
        assign_safe(feats, f"buy_sell_ratio_{period}", buy_vol / sell_vol.replace(0, np.nan))

    bars_per_day = {"1h": 24, "4h": 6, "d1": 1}.get(period, 1)
    log_ret = np.log(feats["close"] / feats["close"].shift(1))
    for d in (7, 14, 30):
        window = d * bars_per_day
        hv = log_ret.rolling(window, min_periods=2).std() * np.sqrt(bars_per_day)
        assign_safe(feats, f"hv_{d}d_{period}", hv)

    # BTC / ETH 短期相关性
    def _find_price(col_candidates):
        for c in df.columns:
            lc = c.lower()
            for cand in col_candidates:
                if cand in lc:
                    return pd.to_numeric(df[c], errors="coerce")
        return None

    btc_price = _find_price(["btc_close", "close_btc", "btcusdt_close", "btc_price"])
    eth_price = _find_price(["eth_close", "close_eth", "ethusdt_close", "eth_price"])
    asset_ret = feats["close"].pct_change(fill_method=None)
    if btc_price is not None:
        btc_ret = btc_price.pct_change(fill_method=None)
        corr = asset_ret.rolling(bars_per_day).corr(btc_ret)
        assign_safe(feats, f"btc_correlation_1h_{period}", corr)
    if eth_price is not None:
        eth_ret = eth_price.pct_change(fill_method=None)
        corr = asset_ret.rolling(bars_per_day).corr(eth_ret)
        assign_safe(feats, f"eth_correlation_1h_{period}", corr)

    range_ = (feats["high"] - feats["low"]).replace(0, np.nan)
    body = (feats["close"] - feats["open"]).abs()
    assign_safe(
        feats, f"upper_wick_ratio_{period}", (feats["high"] - np.maximum(feats["open"], feats["close"])) / range_
    )
    assign_safe(
        feats, f"lower_wick_ratio_{period}", (np.minimum(feats["open"], feats["close"]) - feats["low"]) / range_
    )
    assign_safe(feats, f"body_ratio_{period}", body / range_)

    # === 新增：长影线与低波动突破等结构特征 ===
    upper_long = (feats[f"upper_wick_ratio_{period}"] > 0.6) & (feats[f"body_ratio_{period}"] < 0.3)
    lower_long = (feats[f"lower_wick_ratio_{period}"] > 0.6) & (feats[f"body_ratio_{period}"] < 0.3)
    assign_safe(feats, f"long_upper_shadow_{period}", upper_long.astype(float))
    assign_safe(feats, f"long_lower_shadow_{period}", lower_long.astype(float))

    sma_bbw = _safe_ta(ta.sma, feats[f"bb_width_{period}"], length=20, index=df.index)
    sma_bbw_s = sma_bbw.iloc[:, 0]
    vol_breakout = (feats[f"bb_width_{period}"] > sma_bbw_s * 1.5) & (feats[f"vol_ma_ratio_{period}"] > 1.5)
    assign_safe(feats, f"vol_breakout_{period}", vol_breakout.astype(float))

    assign_safe(feats, f"vol_profile_density_{period}", feats["volume"] / range_)
    assign_safe(
        feats, f"bid_ask_spread_pct_{period}", (feats["high"] - feats["low"]) / feats["close"].replace(0, np.nan)
    )

    returns = feats["close"].pct_change(fill_method=None)
    assign_safe(feats, f"skewness_{period}", returns.rolling(20).skew())
    assign_safe(feats, f"kurtosis_{period}", returns.rolling(20).kurt())

    feats[f"bull_streak_{period}"] = (
        feats["close"]
        .gt(feats["open"])
        .astype(float)
        .groupby(feats["close"].le(feats["open"]).astype(int).cumsum())
        .cumsum()
    )
    feats[f"bear_streak_{period}"] = (
        feats["close"]
        .lt(feats["open"])
        .astype(float)
        .groupby(feats["close"].ge(feats["open"]).astype(int).cumsum())
        .cumsum()
    )

    assign_safe(feats, f"rsi_mul_vol_ma_ratio_{period}", feats[f"rsi_{period}"] * feats[f"vol_ma_ratio_{period}"])
    assign_safe(
        feats,
        f"willr_{period}",
        _safe_ta(ta.willr, feats["high"], feats["low"], feats["close"], length=14, index=df.index),
    )

    macd = _safe_ta(
        ta.macd,
        feats["close"],
        index=df.index,
        cols=["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
    )
    assign_safe(feats, f"macd_{period}", macd["MACD_12_26_9"])
    assign_safe(feats, f"macd_signal_{period}", macd["MACDs_12_26_9"])
    assign_safe(feats, f"macd_hist_{period}", macd["MACDh_12_26_9"])

    assign_safe(feats, f"obv_{period}", _safe_ta(ta.obv, feats["close"], feats["volume"], index=df.index))
    feats[f"obv_delta_{period}"] = feats[f"obv_{period}"].diff()

    st = _safe_ta(
        ta.supertrend,
        feats["high"],
        feats["low"],
        feats["close"],
        index=df.index,
        cols=["SUPERT_7_3.0", "SUPERTd_7_3.0", "SUPERTl_7_3.0", "SUPERTs_7_3.0"],
    )
    assign_safe(feats, f"supertrend_dir_{period}", st.get("SUPERTd_7_3.0", pd.Series(index=df.index, data=np.nan)))

    # ======== CoinGecko 衍生特征 ========
    if "cg_price" in df:
        cg_price = pd.to_numeric(df["cg_price"], errors="coerce")
        orig_close = pd.to_numeric(df["close"], errors="coerce")
        assign_safe(feats, f"price_diff_cg_{period}", orig_close - cg_price)
        assign_safe(
            feats,
            f"price_ratio_cg_{period}",
            orig_close / cg_price.replace(0, np.nan),
        )

    if "cg_market_cap" in df:
        cg_mc = pd.to_numeric(df["cg_market_cap"], errors="coerce")
        assign_safe(feats, f"cg_market_cap_roc_{period}", cg_mc.pct_change(fill_method=None))

    if "cg_total_volume" in df:
        cg_tv = pd.to_numeric(df["cg_total_volume"], errors="coerce")
        assign_safe(feats, f"cg_total_volume_roc_{period}", cg_tv.pct_change(fill_method=None))
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
    return feats


def calc_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """根据 order_book 快照计算买卖盘数量差比率"""
    index = pd.to_datetime(df["timestamp"])
    bid_sum = df["bids"].apply(lambda x: sum(float(b[1]) for b in (x if isinstance(x, list) else json.loads(x))))
    ask_sum = df["asks"].apply(lambda x: sum(float(a[1]) for a in (x if isinstance(x, list) else json.loads(x))))
    denom = (bid_sum + ask_sum).replace(0, np.nan)
    imbalance = (bid_sum - ask_sum) / denom
    df_out = pd.DataFrame({"bid_ask_imbalance": imbalance.values}, index=index)
    df_out.index.name = "open_time"
    return df_out

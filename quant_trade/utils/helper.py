import numpy as np

import pandas as pd
import pandas_ta as ta
import json
import warnings
import logging
from sklearn.preprocessing import RobustScaler
from numba import njit
from quant_trade.utils.soft_clip import soft_clip
from typing import Mapping, Any

logger = logging.getLogger(__name__)


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
        res = pd.DataFrame(
            0,
            index=index,
            columns=cols or ["val"],
            dtype="float64",
        )
    else:
        if isinstance(res, pd.Series):
            res = res.to_frame()
        res = res.astype("float64")
        if cols is not None:
            for c in cols:
                if c not in res.columns:
                    res[c] = np.nan
            res = res[cols]
    return res.sort_index()


def assign_safe(feats: pd.DataFrame, name: str, series):
    """安全地向 feats 赋值，支持传入 Series 或单列 DataFrame。"""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("assign_safe 仅支持单列 DataFrame")
        series = series.iloc[:, 0]
    feats[name] = np.asarray(series, dtype="float64")
    # print(f"{name}: {feats[name].dtype}")


def get_cfg_value(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    """Return ``cfg[key]`` if available and not ``None``, else ``default``."""
    if cfg is None:
        return default
    val = cfg.get(key, default)
    return default if val is None else val


def get_feat(
    raw_dict: Mapping[str, Any] | None,
    std_dict: Mapping[str, Any] | None,
    key: str,
    default: Any | None = None,
) -> Any:
    """Return feature ``key`` from ``raw_dict`` then ``std_dict`` with fallback.

    Parameters
    ----------
    raw_dict : Mapping[str, Any] | None
        原始特征字典。
    std_dict : Mapping[str, Any] | None
        标准化特征字典。
    key : str
        待读取的特征名。
    default : Any, optional
        找不到时返回的默认值，默认为 ``None``。
    """

    if raw_dict is not None:
        val = raw_dict.get(key)
        if val is not None:
            return val
    if std_dict is not None:
        val = std_dict.get(key)
        if val is not None:
            return val
    return default


def calc_mfi_np(high, low, close, volume, window=14):
    """Return Money Flow Ratio and Money Flow Index"""
    if len(high) == 0:
        return np.array([]), np.array([])
    tp = (high + low + close) / 3
    mf = tp * volume
    pmf = np.where(tp > np.roll(tp, 1), mf, 0)
    nmf = np.where(tp < np.roll(tp, 1), mf, 0)
    pmf[0] = 0
    nmf[0] = 0
    sum_pmf = (
        pd.Series(pmf)
        .rolling(window, min_periods=1)
        .sum()
        .to_numpy()
    )
    sum_nmf = (
        pd.Series(nmf)
        .rolling(window, min_periods=1)
        .sum()
        .to_numpy()
    )
    ratio = sum_pmf / (sum_nmf + 1e-12)
    mfi = np.divide(100 * sum_pmf, sum_pmf + sum_nmf + 1e-12)
    return ratio, mfi


def vwap_np(high, low, close, volume, window: int | None = None):
    """Compute VWAP using cumulative sums or rolling window."""
    tp = (np.asarray(high) + np.asarray(low) + np.asarray(close)) / 3
    volume_arr = np.asarray(volume, dtype="float64")
    dollar = tp * volume_arr
    if window is None:
        cum_vol = np.cumsum(volume_arr)
        cum_dollar = np.cumsum(dollar)
    else:
        if window <= 0:
            raise ValueError("window must be positive")
        vol_series = pd.Series(volume_arr)
        dollar_series = pd.Series(dollar)
        cum_vol = vol_series.rolling(window, min_periods=1).sum().to_numpy()
        cum_dollar = dollar_series.rolling(window, min_periods=1).sum().to_numpy()
    vwap = cum_dollar / np.where(cum_vol == 0, np.nan, cum_vol)
    if isinstance(high, (pd.Series, pd.DataFrame)):
        return pd.Series(vwap, index=getattr(high, "index", None))
    return vwap


# === Numba 加速的 VPOC 计算 ===============================================
@njit
def _vpoc_numba(close: np.ndarray, volume: np.ndarray, window: int = 200, bins: int = 24) -> np.ndarray:
    """基于成交量的分布计算近 ``window`` 根的成交密集价格 (VPOC)。"""

    n = len(close)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = 0 if i < window else i - window + 1
        c_slice = close[lo : i + 1]
        v_slice = volume[lo : i + 1]
        c_min = c_slice.min()
        c_max = c_slice.max()
        step = (c_max - c_min) / bins
        if step == 0:
            out[i] = c_slice[0]
            continue
        hist = np.zeros(bins, dtype=np.float64)
        for j in range(c_slice.shape[0]):
            idx = int((c_slice[j] - c_min) / step)
            if idx == bins:
                idx -= 1
            hist[idx] += v_slice[j]
        max_idx = np.argmax(hist)
        out[i] = c_min + (max_idx + 0.5) * step
    return out


def calc_td_sequential(close: pd.Series, lookback: int = 4) -> pd.DataFrame:
    """计算 TD Sequential 买入/卖出计数。

    Parameters
    ----------
    close : pd.Series
        收盘价序列。
    lookback : int, default 4
        与多少根前的收盘价比较，默认与前 4 根比较。

    Returns
    -------
    pd.DataFrame
        包含 ``td_buy_count`` 与 ``td_sell_count`` 两列。
    """

    close = pd.to_numeric(close, errors="coerce")
    shifted = close.shift(lookback)
    buy_cond = (close < shifted) & close.notna() & shifted.notna()
    sell_cond = (close > shifted) & close.notna() & shifted.notna()

    buy_groups = (~buy_cond).astype(int).cumsum()
    sell_groups = (~sell_cond).astype(int).cumsum()

    buy = buy_cond.astype(float).groupby(buy_groups).cumsum()
    sell = sell_cond.astype(float).groupby(sell_groups).cumsum()

    df = pd.DataFrame({"td_buy_count": buy, "td_sell_count": sell})
    df.index = close.index
    return df


def calc_price_channel(high: pd.Series, low: pd.Series, close: pd.Series, *, window: int = 20) -> pd.DataFrame:
    """计算价格通道及位置

    Parameters
    ----------
    high, low, close : pd.Series
        价格序列。
    window : int, default 20
        通道计算窗口，默认使用 20 根K线。

    Returns
    -------
    pd.DataFrame
        包含 ``upper``、``lower``、``channel_pos`` 三列。
    """

    upper = (
        high.fillna(-np.inf)
        .rolling(window, min_periods=1)
        .max()
        .replace(-np.inf, np.nan)
    )
    lower = (
        low.fillna(np.inf)
        .rolling(window, min_periods=1)
        .min()
        .replace(np.inf, np.nan)
    )
    pos = (close - lower) / (upper - lower).replace(0, np.nan)
    df = pd.DataFrame({"upper": upper, "lower": lower, "channel_pos": pos})
    df.index = close.index
    return df


def calc_support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """计算支撑阻力水平及突破标记"""

    support = (
        low.fillna(np.inf)
        .rolling(window, min_periods=1)
        .min()
        .replace(np.inf, np.nan)
    )
    resistance = (
        high.fillna(-np.inf)
        .rolling(window, min_periods=1)
        .max()
        .replace(-np.inf, np.nan)
    )

    break_support = close < support.shift(1)
    break_resistance = close > resistance.shift(1)

    df = pd.DataFrame(
        {
            "support": support,
            "resistance": resistance,
            "break_support": break_support.astype(float),
            "break_resistance": break_resistance.astype(float),
        }
    )
    df.index = close.index
    return df


def calc_rsi_divergence(close: pd.Series, rsi: pd.Series) -> pd.DataFrame:
    """计算 RSI 背离

    当价格创新高而 RSI 未创新高时，视为空头背离；
    当价格创新低而 RSI 未创新低时，视为多头背离。

    Parameters
    ----------
    close : pd.Series
        收盘价序列。
    rsi : pd.Series
        RSI 指标序列。

    Returns
    -------
    pd.DataFrame
        包含 ``bull`` 与 ``bear`` 两列，分别代表多头/空头背离标记。
    """

    window = 14
    prev_high = close.shift(1).rolling(window, min_periods=1).max()
    prev_low = close.shift(1).rolling(window, min_periods=1).min()
    prev_rsi_high = rsi.shift(1).rolling(window, min_periods=1).max()
    prev_rsi_low = rsi.shift(1).rolling(window, min_periods=1).min()

    bear = (close > prev_high) & (rsi <= prev_rsi_high)
    bull = (close < prev_low) & (rsi >= prev_rsi_low)

    df = pd.DataFrame({"bull": bull.astype(float), "bear": bear.astype(float)})
    df.index = close.index
    return df


def calc_features_raw(
    df: pd.DataFrame,
    period: str,
    *,
    symbol: str | None = None,
    long_window: int = 30,
    vwap_window: int | None = None,
) -> pd.DataFrame | None:
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    df = df.loc[~df.index.duplicated()].copy()
    if not df.index.is_monotonic_increasing:
        warnings.warn("Index is not monotonic increasing, resorting by index")
        df = df.sort_index()
    # 确保时间顺序正确，避免 VWAP 等指标计算异常
    feats = pd.DataFrame(index=df.index)
    feats.sort_index(inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        feats[col] = np.full(len(feats), np.nan, dtype="float64")

    if symbol is not None and len(feats) < long_window:
        logger.debug("%s < %s rows -> skip", symbol, long_window)
        return None

    def _check_index(name: str):
        if not feats.index.is_monotonic_increasing:
            logger.debug("Index broke monotonicity before %s", name)

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
    if "social_sentiment" in df:
        ss = df["social_sentiment"].astype(float).ffill()
        assign_safe(feats, "social_sentiment", ss)
    if "funding_rate" in df:
        assign_safe(feats, "funding_rate", df["funding_rate"].astype(float).ffill())
        _check_index("fr_ema")
        fr_ema = _safe_ta(ta.ema, feats["funding_rate"], length=24, index=df.index)
        fr_ema_s = fr_ema.iloc[:, 0]
        assign_safe(feats, f"funding_rate_anom_{period}", (feats["funding_rate"] - fr_ema_s))

    _check_index("ema_short")
    ema_short = _safe_ta(ta.ema, feats["close"], length=10, index=df.index)
    ema_short_s = ema_short.iloc[:, 0]
    if ema_short_s.isna().all():
        ema_short_s = feats["close"].ewm(span=10, adjust=False).mean()

    _check_index("ema_long")
    ema_long = _safe_ta(ta.ema, feats["close"], length=50, index=df.index)
    ema_long_s = ema_long.iloc[:, 0]
    if ema_long_s.isna().all():
        ema_long_s = feats["close"].ewm(span=50, adjust=False).mean()

    assign_safe(feats, f"ema_diff_{period}", ema_short_s - ema_long_s)
    _check_index("sma")
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
    _check_index("rsi")
    assign_safe(feats, f"rsi_{period}", _safe_ta(ta.rsi, feats["close"], length=14, index=df.index))
    assign_safe(feats, f"rsi_fast_{period}", _safe_ta(ta.rsi, feats["close"], length=7, index=df.index))
    feats[f"rsi_slope_{period}"] = feats[f"rsi_{period}"].diff()

    div = calc_rsi_divergence(feats["close"], feats[f"rsi_{period}"])
    assign_safe(feats, f"rsi_bull_div_{period}", div["bull"])
    assign_safe(feats, f"rsi_bear_div_{period}", div["bear"])
    _check_index("atr")
    atr = _safe_ta(ta.atr, feats["high"], feats["low"], feats["close"], length=14, index=df.index)
    atr_s = atr.iloc[:, 0]
    assign_safe(feats, f"atr_pct_{period}", atr_s.div(feats["close"], axis=0))
    feats[f"atr_chg_{period}"] = feats[f"atr_pct_{period}"].diff()

    # === Classic Pivot ==================================================
    pivot = (feats["high"] + feats["low"] + feats["close"]) / 3
    feats[f"pivot_{period}"] = pivot
    feats[f"pivot_r1_{period}"] = 2 * pivot - feats["low"]
    feats[f"pivot_s1_{period}"] = 2 * pivot - feats["high"]
    feats[f"close_vs_pivot_{period}"] = (feats["close"] - pivot) / feats["close"]

    # === 简版 VPOC（近 200 根） =========================================
    vpoc = _vpoc_numba(
        feats["close"].to_numpy(dtype=np.float64),
        feats["volume"].to_numpy(dtype=np.float64),
        window=200,
        bins=24,
    )
    feats[f"vpoc_{period}"] = vpoc
    feats[f"close_vs_vpoc_{period}"] = (
        feats["close"] - feats[f"vpoc_{period}"]
    ) / feats["close"]

    _check_index("adx")
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
    _check_index("cci")
    assign_safe(
        feats, f"cci_{period}", _safe_ta(ta.cci, feats["high"], feats["low"], feats["close"], length=14, index=df.index)
    )
    feats[f"cci_delta_{period}"] = feats[f"cci_{period}"].diff().fillna(0)
    _check_index("mfi")
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
    _check_index("vwap")
    assign_safe(
        feats,
        f"vwap_{period}",
        vwap_np(
            feats["high"],
            feats["low"],
            feats["close"],
            feats["volume"],
            window=vwap_window,
        ),
    )
    _check_index("stoch")
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

    stoch_fast = _safe_ta(
        ta.stoch,
        feats["high"],
        feats["low"],
        feats["close"],
        k=9,
        d=3,
        smooth_k=3,
        index=df.index,
        cols=["STOCHk_9_3_3", "STOCHd_9_3_3"],
    )
    stoch_fast = stoch_fast.reindex(df.index)
    assign_safe(feats, f"stoch_fast_{period}", stoch_fast.get("STOCHk_9_3_3"))

    _check_index("bbands")
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

    _check_index("kc")
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

    _check_index("ichimoku")
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

    _check_index("donchian")
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

    _check_index("vol_roc")
    assign_safe(feats, f"vol_roc_{period}", _safe_ta(ta.roc, feats["volume"], length=5, index=df.index))
    _check_index("sma_vol_short")
    sma_vol_short = _safe_ta(ta.sma, feats["volume"], length=10, index=df.index).iloc[:, 0]
    assign_safe(
        feats,
        f"vol_ma_ratio_{period}",
        feats["volume"] / sma_vol_short.replace(0, np.nan),
    )
    _check_index("sma_vol_long")
    sma_vol_long = feats["volume"].rolling(long_window, min_periods=1).mean()
    vol_ma_ratio_long = feats["volume"].div(sma_vol_long).fillna(0)
    assign_safe(feats, f"vol_ma_ratio_long_{period}", vol_ma_ratio_long)

    if "taker_buy_base" in df:
        buy_vol = pd.to_numeric(df["taker_buy_base"], errors="coerce")
        sell_vol = feats["volume"] - buy_vol
        assign_safe(feats, f"buy_sell_ratio_{period}", buy_vol / sell_vol.replace(0, np.nan))

    bars_per_day = {"1h": 24, "4h": 6, "d1": 1}.get(period, 1)
    log_ret = np.log(feats["close"] / feats["close"].shift(1))
    _check_index("hv")
    for d in (7, 14, 30):
        window = d * bars_per_day
        hv = log_ret.rolling(window, min_periods=2).std() * np.sqrt(bars_per_day)
        assign_safe(feats, f"hv_{d}d_{period}", hv)

    # 价格通道位置
    _check_index("price_channel")
    channel = calc_price_channel(feats["high"], feats["low"], feats["close"], window=20)
    assign_safe(feats, f"channel_pos_{period}", channel["channel_pos"])

    sr = calc_support_resistance(feats["high"], feats["low"], feats["close"], window=20)
    assign_safe(feats, f"support_level_{period}", sr["support"])
    assign_safe(feats, f"resistance_level_{period}", sr["resistance"])
    assign_safe(feats, f"break_support_{period}", sr["break_support"])
    assign_safe(feats, f"break_resistance_{period}", sr["break_resistance"])

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
    _check_index("correlation")
    if btc_price is not None:
        btc_ret = btc_price.pct_change(fill_method=None)
        corr = asset_ret.rolling(bars_per_day, min_periods=1).corr(btc_ret)
        assign_safe(feats, f"btc_correlation_1h_{period}", corr)
    if eth_price is not None:
        eth_ret = eth_price.pct_change(fill_method=None)
        corr = asset_ret.rolling(bars_per_day, min_periods=1).corr(eth_ret)
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

    _check_index("sma_bbw")
    sma_bbw = _safe_ta(ta.sma, feats[f"bb_width_{period}"], length=20, index=df.index)
    sma_bbw_s = sma_bbw.iloc[:, 0]
    # 布林带宽度明显高于均值时视为放量突破
    vol_breakout = (
        feats[f"bb_width_{period}"] > sma_bbw_s * 1.2
    ) & (feats[f"vol_ma_ratio_{period}"] > 1.2)
    assign_safe(feats, f"vol_breakout_{period}", vol_breakout.astype(float))
    # 布林带宽度低于均值 70% 视为收敛
    bb_squeeze = feats[f"bb_width_{period}"] < sma_bbw_s * 0.7
    assign_safe(feats, f"bb_squeeze_{period}", bb_squeeze.astype(float))

    range_dens = (feats["high"] - feats["low"]).abs().clip(lower=1e-6)
    density = feats["volume"] / range_dens
    assign_safe(feats, f"vol_profile_density_{period}", soft_clip(density))
    assign_safe(
        feats, f"bid_ask_spread_pct_{period}", (feats["high"] - feats["low"]) / feats["close"].replace(0, np.nan)
    )

    returns = feats["close"].pct_change(fill_method=None)
    assign_safe(feats, f"skewness_{period}", returns.rolling(20, min_periods=1).skew())
    assign_safe(feats, f"kurtosis_{period}", returns.rolling(20, min_periods=1).kurt())

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

    td = calc_td_sequential(feats["close"])
    assign_safe(feats, f"td_buy_count_{period}", td["td_buy_count"])
    assign_safe(feats, f"td_sell_count_{period}", td["td_sell_count"])

    assign_safe(feats, f"rsi_mul_vol_ma_ratio_{period}", feats[f"rsi_{period}"] * feats[f"vol_ma_ratio_{period}"])
    _check_index("willr")
    assign_safe(
        feats,
        f"willr_{period}",
        _safe_ta(ta.willr, feats["high"], feats["low"], feats["close"], length=14, index=df.index),
    )

    _check_index("macd")
    macd = _safe_ta(
        ta.macd,
        feats["close"],
        index=df.index,
        cols=["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
    )
    assign_safe(feats, f"macd_{period}", macd["MACD_12_26_9"])
    assign_safe(feats, f"macd_signal_{period}", macd["MACDs_12_26_9"])
    assign_safe(feats, f"macd_hist_{period}", macd["MACDh_12_26_9"])

    _check_index("obv")
    assign_safe(feats, f"obv_{period}", _safe_ta(ta.obv, feats["close"], feats["volume"], index=df.index))
    feats[f"obv_delta_{period}"] = feats[f"obv_{period}"].diff()

    _check_index("supertrend")
    st = _safe_ta(
        ta.supertrend,
        feats["high"],
        feats["low"],
        feats["close"],
        length=10,
        multiplier=3.0,
        index=df.index,
        cols=["SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0"],
    )
    assign_safe(feats, f"supertrend_dir_{period}", st.get("SUPERTd_10_3.0", pd.Series(index=df.index, data=np.nan)))

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

    # ======== CM 链上指标 ========
    if "AdrActCnt" in df:
        active = pd.to_numeric(df["AdrActCnt"], errors="coerce")
        assign_safe(
            feats,
            f"active_addr_roc_{period}",
            active.pct_change(fill_method=None),
        )

    if "AdrNewCnt" in df:
        new_addr = pd.to_numeric(df["AdrNewCnt"], errors="coerce")
        assign_safe(
            feats,
            f"new_addr_roc_{period}",
            new_addr.pct_change(fill_method=None),
        )

    if "TxCnt" in df:
        tx_cnt = pd.to_numeric(df["TxCnt"], errors="coerce")
        assign_safe(
            feats,
            f"tx_count_roc_{period}",
            tx_cnt.pct_change(fill_method=None),
        )

    if "CapMrktCurUSD" in df and "CapRealUSD" in df:
        cap_mkt = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
        cap_real = pd.to_numeric(df["CapRealUSD"], errors="coerce")
        assign_safe(
            feats,
            f"mvrv_ratio_{period}",
            cap_mkt / cap_real.replace(0, np.nan),
        )

    if "FeeTotUSD" in df:
        fee_tot = pd.to_numeric(df["FeeTotUSD"], errors="coerce")
        assign_safe(feats, f"fee_tot_roc_{period}", fee_tot.pct_change(fill_method=None))

    if "RevHashUSD" in df:
        rev_hash = pd.to_numeric(df["RevHashUSD"], errors="coerce")
        assign_safe(feats, f"rev_hash_roc_{period}", rev_hash.pct_change(fill_method=None))

    if "IssTotUSD" in df:
        iss_tot = pd.to_numeric(df["IssTotUSD"], errors="coerce")
        assign_safe(feats, f"iss_tot_roc_{period}", iss_tot.pct_change(fill_method=None))

    if "SplyCur" in df:
        sply_cur = pd.to_numeric(df["SplyCur"], errors="coerce")
        assign_safe(feats, f"sply_cur_roc_{period}", sply_cur.pct_change(fill_method=None))

    if "SplyAct1Yr" in df:
        sply_act = pd.to_numeric(df["SplyAct1Yr"], errors="coerce")
        assign_safe(feats, f"sply_act_1yr_roc_{period}", sply_act.pct_change(fill_method=None))
        if "SplyCur" in df:
            assign_safe(feats, f"sply_act_pct_{period}", sply_act / sply_cur.replace(0, np.nan))

    if "HashRate" in df:
        hash_rate = pd.to_numeric(df["HashRate"], errors="coerce")
        assign_safe(feats, f"hash_rate_roc_{period}", hash_rate.pct_change(fill_method=None))

    if "DiffMean" in df:
        diff_mean = pd.to_numeric(df["DiffMean"], errors="coerce")
        assign_safe(feats, f"diff_mean_roc_{period}", diff_mean.pct_change(fill_method=None))

    if "HashRate" in df and "DiffMean" in df:
        assign_safe(feats, f"hashrate_difficulty_ratio_{period}", hash_rate / diff_mean.replace(0, np.nan))

    if "FeeTotUSD" in df and "RevHashUSD" in df:
        assign_safe(feats, f"fee_rev_ratio_{period}", fee_tot / rev_hash.replace(0, np.nan))

    if "FeeTotUSD" in df and "IssTotUSD" in df:
        assign_safe(feats, f"fee_iss_ratio_{period}", fee_tot / iss_tot.replace(0, np.nan))

    if "IssTotUSD" in df and "RevHashUSD" in df:
        assign_safe(feats, f"iss_rev_ratio_{period}", iss_tot / rev_hash.replace(0, np.nan))

    if symbol is not None:
        null_ratio = feats.isnull().all(axis=1).mean()
        if null_ratio > 0.5:
            logger.debug("%s too many NaN rows -> skip", symbol)
            return None

    return feats


def calc_features_full(df: pd.DataFrame, period: str) -> pd.DataFrame:
    feats = calc_features_raw(df, period)

    num_cols = feats.select_dtypes("number").columns
    feats[num_cols] = RobustScaler(quantile_range=(1, 99)).fit_transform(feats[num_cols])

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

# -*- coding: utf-8 -*-
"""
FeatureEngineer v2.3-patch1 (External indicators removed)  (2025-06-03)
==================================================================
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from sqlalchemy import create_engine, text, bindparam, inspect
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sklearn.metrics import mutual_info_score
from scipy import stats
import json

# 不再 import calc_features_full，而改为：
from quant_trade.utils.helper import (
    calc_features_raw,
    calc_order_book_features,
)  # pylint: disable=import-error

# Robust-z 参数持久化工具
from quant_trade.utils.robust_scaler import (
    compute_robust_z_params,
    save_scaler_params_to_json,
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)

from quant_trade.utils.feature_health import health_check, apply_health_check_df
from quant_trade.utils.soft_clip import soft_clip
from quant_trade.logging import get_logger

logger = get_logger(__name__)

# Future-related columns to drop for leakage prevention
FUTURE_COLS = [
    "future_volatility",
    "future_volatility_1h",
    "future_volatility_4h",
    "future_volatility_d1",
    "future_max_rise",
    "future_max_rise_1h",
    "future_max_rise_4h",
    "future_max_rise_d1",
    "future_max_drawdown",
    "future_max_drawdown_1h",
    "future_max_drawdown_4h",
    "future_max_drawdown_d1",
    "target",
    "target_1h",
    "target_4h",
    "target_d1",
]

# 基础列，合并批次时保持在前，且不参与归一化
BASE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "symbol",
    "target",
] + FUTURE_COLS

# FUTURE_COLS 中包含 "target"，拼接后需去重
BASE_COLS = list(dict.fromkeys(BASE_COLS))


def calc_cross_features(
    df1h: pd.DataFrame, df4h: pd.DataFrame, df1d: pd.DataFrame
) -> pd.DataFrame:
    """生成跨周期特征并返回合并后的 DataFrame."""

    f1h = df1h.copy()
    f4h = df4h.copy()
    f1d = df1d.copy()

    for df in (f1h, f4h, f1d):
        if "open_time" in df.columns:
            df.reset_index(drop=True, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")
        # 确保 open_time 为 datetime 类型，避免与 Timedelta 运算时报错
        df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")

    if "social_sentiment" in f1h.columns:
        ss = f1h["social_sentiment"].astype(float)
        f1h["social_sentiment_1h"] = ss
        f1h["social_sentiment_4h"] = ss.rolling(4, min_periods=1).mean()
        f1h = f1h.drop(columns=["social_sentiment"])

    f1h = f1h.rename(columns={"close": "close_1h"})
    f4h = f4h.rename(columns={"close": "close_4h"})
    f1d = f1d.rename(columns={"close": "close_d1"})

    f4h["close_time_4h"] = (
        f4h["open_time"] + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
    )
    f1d["close_time_d1"] = (
        f1d["open_time"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    f4h = f4h.drop(columns=["open_time"])
    f1d = f1d.drop(columns=["open_time"])

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

    merged["close_spread_1h_4h"] = merged["close_1h"] - merged["close_4h"]
    merged["close_spread_1h_d1"] = merged["close_1h"] - merged["close_d1"]
    merged["ma_ratio_1h_4h"] = merged["sma_10_1h"] / merged["sma_10_4h"].replace(
        0, np.nan
    )
    merged["ma_ratio_1h_d1"] = merged["sma_10_1h"] / merged["sma_10_d1"].replace(
        0, np.nan
    )
    merged["ma_ratio_4h_d1"] = merged["sma_10_4h"] / merged["sma_10_d1"].replace(
        0, np.nan
    )
    merged["atr_pct_ratio_1h_4h"] = merged["atr_pct_1h"] / merged["atr_pct_4h"].replace(
        0, np.nan
    )
    merged["bb_width_ratio_1h_4h"] = merged["bb_width_1h"] / merged["bb_width_4h"].replace(
        0, np.nan
    )
    merged["rsi_diff_1h_4h"] = merged["rsi_1h"] - merged["rsi_4h"]
    merged["rsi_diff_1h_d1"] = merged["rsi_1h"] - merged["rsi_d1"]
    merged["rsi_diff_4h_d1"] = merged["rsi_4h"] - merged["rsi_d1"]
    merged["macd_hist_diff_1h_4h"] = merged["macd_hist_1h"] - merged["macd_hist_4h"]
    merged["macd_hist_diff_1h_d1"] = merged["macd_hist_1h"] - merged["macd_hist_d1"]
    merged["macd_hist_4h_mul_bb_width_1h"] = merged["macd_hist_4h"] * merged["bb_width_1h"]
    merged["rsi_1h_mul_vol_ma_ratio_4h"] = merged["rsi_1h"] * merged["vol_ma_ratio_4h"]
    merged["macd_hist_1h_mul_bb_width_4h"] = merged["macd_hist_1h"] * merged["bb_width_4h"]
    merged["vol_ratio_1h_4h"] = merged["vol_ma_ratio_1h"] / merged["vol_ma_ratio_4h"].replace(
        0, np.nan
    )
    merged["vol_ratio_4h_d1"] = merged["vol_ma_ratio_4h"] / merged["vol_ma_ratio_d1"].replace(
        0, np.nan
    )
    merged["ma_ratio_5_20"] = merged["sma_5_1h"] / merged["sma_20_1h"].replace(0, np.nan)

    return merged


def calc_cross_features_v2(
    df1h: pd.DataFrame, df4h: pd.DataFrame, df1d: pd.DataFrame
) -> pd.DataFrame:
    """改进版跨周期特征生成函数。"""

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
            merged[f"close_spread_{a}_{b}"] = merged[f"close_{a}"] - merged[f"close_{b}"]
            if f"sma_10_{a}" in merged.columns and f"sma_10_{b}" in merged.columns:
                merged[f"ma_ratio_{a}_{b}"] = merged[f"sma_10_{a}"] / merged[f"sma_10_{b}"].replace(0, np.nan)
            if f"atr_pct_{a}" in merged.columns and f"atr_pct_{b}" in merged.columns:
                merged[f"atr_pct_ratio_{a}_{b}"] = merged[f"atr_pct_{a}"] / merged[f"atr_pct_{b}"].replace(0, np.nan)
            if f"bb_width_{a}" in merged.columns and f"bb_width_{b}" in merged.columns:
                merged[f"bb_width_ratio_{a}_{b}"] = merged[f"bb_width_{a}"] / merged[f"bb_width_{b}"].replace(0, np.nan)
            if f"rsi_{a}" in merged.columns and f"rsi_{b}" in merged.columns:
                merged[f"rsi_diff_{a}_{b}"] = merged[f"rsi_{a}"] - merged[f"rsi_{b}"]
            if f"macd_hist_{a}" in merged.columns and f"macd_hist_{b}" in merged.columns:
                merged[f"macd_hist_diff_{a}_{b}"] = merged[f"macd_hist_{a}"] - merged[f"macd_hist_{b}"]
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


class FeatureEngineer:
    """多周期特征工程生成器 (1h → 4h → d1)。

    主要修复 / 改进
    ----------------
    1. 无未来数据泄漏的 add_up_down_targets。
    2. 从 feature_cols.txt 动态读取所有待标准化特征，无需手写列表。
    3. 支持 Robust-z 参数持久化：训练时算出 p1/p99/mean/std 并保存；推理时直接加载。
    4. 过滤掉非数值列，只对数值列做 Robust-z。
    5. _add_missing_flags 中一次性 concat，避免 DataFrame 碎片化。
    """

    def __init__(self, config_path: str | os.PathLike = "utils/config.yaml", *, include_order_book: bool = False) -> None:
        path = Path(config_path)
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).resolve().parent / path
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        # MySQL 连接配置
        mysql_cfg = self.cfg.get("mysql", {})
        self.engine = create_engine(
            f"mysql+pymysql://{mysql_cfg.get('user', 'root')}:{os.getenv('MYSQL_PASSWORD', mysql_cfg.get('password', ''))}@"
            f"{mysql_cfg.get('host', 'localhost')}:{mysql_cfg.get('port', 3306)}/"
            f"{mysql_cfg.get('database', 'quant_trading')}?charset={mysql_cfg.get('charset', 'utf8mb4')}"
        )

        # Feature-engineering 配置
        fe_cfg = self.cfg.get("feature_engineering", {})
        self.topn: int = int(fe_cfg.get("topn", 30))
        self.include_order_book: bool = bool(fe_cfg.get("include_order_book", include_order_book))
        self.feature_cols_path: Path = Path(
            fe_cfg.get("feature_cols_path", "data/merged/feature_cols.txt")
        )
        self.merged_table_path: Path = Path(
            fe_cfg.get("merged_table_path", "data/merged/merged_table.csv")
        )

        # Robust-z 模式：train 或 inference
        self.mode: str = fe_cfg.get("mode", "train")

        # 参考币种
        self.btc_symbol: str = fe_cfg.get("btc_symbol", "BTCUSDT")
        self.eth_symbol: str = fe_cfg.get("eth_symbol", "ETHUSDT")
        self.rise_transform: str = fe_cfg.get("rise_transform", "none")
        self.boxcox_lambda_path: Path = Path(
            fe_cfg.get("boxcox_lambda_path", "scalers/rise_boxcox_lambda.json")
        )

        self.period_cfg: dict[str, dict[str, float | int]] = {
            "1h": {"q_low": 0.25, "q_up": 0.75, "base_n": 3, "vol_window": 24},
            "4h": {"q_low": 0.30, "q_up": 0.70, "base_n": 12, "vol_window": 96},
            # 平滑 future_max_drawdown_d1，默认窗口 3
            "d1": {
                "q_low": 0.35,
                "q_up": 0.65,
                "base_n": 72,
                "vol_window": 576,
                "smooth_window": 3,
            },
        }

        self._kl_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        # 保存/加载剪裁+缩放参数的 JSON 路径
        self.scaler_path: Path = Path(
            fe_cfg.get("scaler_path", "scalers/all_features_scaler.json")
        )

        # 从 feature_cols.txt 自动读取所有待标准化列；若文件不存在，先设为空列表
        if self.feature_cols_path.is_file():
            lines = self.feature_cols_path.read_text(encoding="utf-8").splitlines()
            cols = [c.strip() for c in lines if c.strip()]
            self.feature_cols_all: List[str] = [c for c in cols if c not in BASE_COLS]
        else:
            self.feature_cols_all = []

        cm_cfg = self.cfg.get("coinmetrics", {})
        self.cm_metrics: List[str] = cm_cfg.get("metrics", [])

        if self.mode != "train" and self.rise_transform == "boxcox":
            if self.boxcox_lambda_path.is_file():
                with open(self.boxcox_lambda_path, "r", encoding="utf-8") as f:
                    self.boxcox_lambda = json.load(f)
            else:
                self.boxcox_lambda = {}
        else:
            self.boxcox_lambda = {}

    def add_up_down_targets(
        self,
        df: pd.DataFrame,
        period_cfg: dict[str, dict[str, float | int]] | None = None,
        smooth_alpha: float = 0.2,
        classification_mode: str = "three",
    ) -> pd.DataFrame:
        """根据 `period_cfg` 为多周期生成涨跌标签。

        Parameters
        ----------
        df : DataFrame
            源数据，需包含 ``close`` 和 ``symbol`` 列。
        period_cfg : dict, optional
            各周期参数，形如 ``{"1h": {"q_low": 0.25, "q_up": 0.75, "base_n": 3}}``。
            可在 ``d1`` 区块加入 ``smooth_window``，对
            ``future_max_drawdown_d1`` 进行 ``rolling().max()`` 平滑。
            若为 ``None``，则使用默认配置。
        smooth_alpha : float, default 0.2
            计算未来收益时的 EWMA 衰减系数。
        classification_mode : str, default "three"
            当前仅支持 ``"three"``，即生成 0/1/2 三分类标签。
        """

        if period_cfg is None:
            period_cfg = {
                "1h": {"q_low": 0.25, "q_up": 0.75, "base_n": 3, "vol_window": 24},
                "4h": {"q_low": 0.30, "q_up": 0.70, "base_n": 12, "vol_window": 96},
                "d1": {"q_low": 0.35, "q_up": 0.65, "base_n": 72, "vol_window": 576},
            }

        results = []
        if not hasattr(self, "boxcox_lambda"):
            self.boxcox_lambda = {}
        for sym, g in df.groupby("symbol", group_keys=False):
            close = g["close"]
            for p, cfg in period_cfg.items():
                base_n = int(cfg.get("base_n", 3))
                q_low = float(cfg.get("q_low", 0.25))
                q_up = float(cfg.get("q_up", 0.75))

                future_prices = pd.concat(
                    [close.shift(-i) for i in range(1, base_n + 1)], axis=1
                )
                fut_hi = future_prices.max(axis=1)
                fut_lo = future_prices.min(axis=1)

                up_ret = fut_hi / close - 1
                down_ret = fut_lo / close - 1
                if p == "d1":
                    window = int(cfg.get("smooth_window", 1))
                    if window > 1:
                        down_ret = down_ret.rolling(window, min_periods=1).max()

                vol_window = int(cfg.get("vol_window", base_n))
                min_periods = int(cfg.get("min_periods", 1))

                up_thr_roll = (
                    up_ret.rolling(vol_window, min_periods=min_periods)
                    .quantile(q_up)
                    .shift(1)
                )
                up_thr_exp = (
                    up_ret.expanding(min_periods).quantile(q_up).shift(1)
                )
                up_thr = up_thr_roll.combine_first(up_thr_exp)

                down_thr_roll = (
                    down_ret.rolling(vol_window, min_periods=min_periods)
                    .quantile(q_low)
                    .shift(1)
                )
                down_thr_exp = (
                    down_ret.expanding(min_periods).quantile(q_low).shift(1)
                )
                down_thr = down_thr_roll.combine_first(down_thr_exp)

                if (down_thr > up_thr).dropna().any():
                    raise ValueError(
                        f"Invalid thresholds for {p}: down > up at some points"
                    )

                if classification_mode != "three":
                    raise ValueError("Only 'three' classification_mode is supported")

                target = np.where(up_ret >= up_thr, 2, np.where(down_ret <= down_thr, 0, 1))

                g[f"target_{p}"] = target.astype(float)
                g[f"future_volatility_{p}"] = close.pct_change().rolling(base_n).std().shift(-base_n)
                rise_vals = up_ret
                if self.rise_transform == "log":
                    rise_vals = np.log1p(up_ret)
                elif self.rise_transform == "boxcox":
                    arr = up_ret + 1.0
                    arr[arr <= 0] = np.nanmin(arr[arr > 0]) if np.any(arr > 0) else 1e-6
                    valid = arr[~np.isnan(arr)]
                    if p not in self.boxcox_lambda:
                        _, lmbda = stats.boxcox(valid)
                        self.boxcox_lambda[p] = float(lmbda)
                    else:
                        lmbda = self.boxcox_lambda[p]
                    rise_vals = np.full_like(arr, np.nan, dtype=float)
                    rise_vals[~np.isnan(arr)] = stats.boxcox(valid, lmbda)
                g[f"future_max_rise_{p}"] = rise_vals
                g[f"future_max_drawdown_{p}"] = down_ret

                g.loc[g.tail(base_n).index, [f"target_{p}", f"future_volatility_{p}"]] = np.nan

            results.append(g)

        out = pd.concat(results).sort_index()

        for p in period_cfg:
            col = f"target_{p}"
            if col in out.columns:
                cnt = out[col].value_counts(dropna=True)
                ratio = (cnt / cnt.sum()).round(4)
                logger.info("%s 标签分布: %s 比例: %s", p, cnt.to_dict(), ratio.to_dict())

        rename_map = {
            "target": "target_1h",
            "future_volatility": "future_volatility_1h",
            "future_max_rise": "future_max_rise_1h",
            "future_max_drawdown": "future_max_drawdown_1h",
        }
        for base, new in rename_map.items():
            if new in out.columns:
                out[base] = out[new]

        return out

    def get_symbols(
        self, intervals: tuple[str, str, str] = ("1h", "4h", "d1")
    ) -> List[str]:
        """返回同时拥有 intervals 三周期数据的 symbol 列表。"""
        symbol_sets: list[set[str]] = []
        for itv in intervals:
            df = pd.read_sql(
                "SELECT DISTINCT symbol FROM klines WHERE `interval`=%s",
                self.engine,
                params=(itv,),
            )
            symbol_sets.append(set(df["symbol"]))
        return sorted(set.intersection(*symbol_sets))

    def _load_klines_raw(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        key = (symbol, interval)
        if key not in self._kl_cache:
            sql = (
                "SELECT * FROM klines WHERE symbol=%s AND `interval`=%s ORDER BY open_time"
            )
            df = pd.read_sql(
                sql,
                self.engine,
                parse_dates=["open_time", "close_time"],
                params=(symbol, interval),
            )
            if df.empty:
                self._kl_cache[key] = None
            else:
                self._kl_cache[key] = df.set_index("open_time").sort_index()
        cached = self._kl_cache.get(key)
        return None if cached is None else cached.copy()

    def load_klines_db(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """读取指定 symbol/interval 的 K 线，并附加 btc/eth 收盘价列."""
        df = self._load_klines_raw(symbol, interval)
        if df is None:
            return None

        out = df.reset_index()

        if symbol != self.btc_symbol:
            btc_df = self._load_klines_raw(self.btc_symbol, interval)
            if btc_df is not None:
                btc_df = btc_df[["close"]].rename(columns={"close": "btc_close"}).reset_index()
                out = pd.merge_asof(
                    out.sort_values("open_time"),
                    btc_df.sort_values("open_time"),
                    on="open_time",
                    direction="backward",
                )

        if symbol != self.eth_symbol:
            eth_df = self._load_klines_raw(self.eth_symbol, interval)
            if eth_df is not None:
                eth_df = eth_df[["close"]].rename(columns={"close": "eth_close"}).reset_index()
                out = pd.merge_asof(
                    out.sort_values("open_time"),
                    eth_df.sort_values("open_time"),
                    on="open_time",
                    direction="backward",
                )

        # merge cm_onchain_metrics
        try:
            if self.cm_metrics:
                q = text(
                    "SELECT timestamp, metric, value FROM cm_onchain_metrics "
                    "WHERE symbol=:sym AND metric IN :metrics ORDER BY timestamp"
                ).bindparams(bindparam("metrics", expanding=True))
                cm_df = pd.read_sql(
                    q,
                    self.engine,
                    params={"sym": symbol, "metrics": self.cm_metrics},
                    parse_dates=["timestamp"],
                )
            else:
                cm_df = None
        except SQLAlchemyError:  # pragma: no cover - optional table
            logger.exception("load cm_onchain_metrics failed")
            cm_df = None

        if cm_df is not None and not cm_df.empty:
            cm_df = (
                cm_df.pivot(index="timestamp", columns="metric", values="value")
                .reset_index()
                .rename(columns={"timestamp": "open_time"})
            )
            out = pd.merge_asof(
                out.sort_values("open_time"),
                cm_df.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            for m in self.cm_metrics:
                out[m] = None

        # merge social sentiment
        try:
            s_df = pd.read_sql(
                "SELECT date, score FROM social_sentiment ORDER BY date",
                self.engine,
                parse_dates=["date"],
            )
        except SQLAlchemyError:  # pragma: no cover - optional table
            logger.exception("load social_sentiment failed")
            s_df = None

        if s_df is not None and not s_df.empty:
            s_df = s_df.rename(columns={"date": "open_time", "score": "social_sentiment"})
            out = pd.merge_asof(
                out.sort_values("open_time"),
                s_df.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            out["social_sentiment"] = None

        return out.set_index("open_time").sort_index()

    def load_order_book(self, symbol: str) -> Optional[pd.DataFrame]:
        """读取指定合约的 order_book 快照"""
        sql = "SELECT * FROM order_book WHERE symbol=%s ORDER BY timestamp"
        df = pd.read_sql(sql, self.engine, params=(symbol,), parse_dates=["timestamp"])
        if df.empty:
            return None
        return df

    def _add_missing_flags(
        self, df: pd.DataFrame, feat_cols: list
    ) -> tuple[pd.DataFrame, list]:
        """Forward fill, drop极稀疏列, 并追加缺失标记。"""

        missing_ratio = df[feat_cols].isna().mean()

        # ----- 先为所有特征生成缺失标记 -----
        flags_df = df[feat_cols].isna().astype(int)
        flags_df.columns = [f"{col}_isnan" for col in feat_cols]


        # ----- 不再因缺失率高而删除列，保留所有特征 -----

        df_filled = df.copy()
        if "open_time" in df_filled.columns:
            if "symbol" in df_filled.columns:
                df_filled = df_filled.sort_values(["symbol", "open_time"])
            else:
                df_filled = df_filled.sort_values("open_time")
        df_filled[feat_cols] = (
            df_filled.groupby("symbol")[feat_cols].ffill()
            if "symbol" in df_filled.columns
            else df_filled[feat_cols].ffill()
        )
        df_filled = df_filled.sort_index()

        df_out = pd.concat([df_filled, flags_df], axis=1)
        return df_out, feat_cols

    def _finalize_batch(
        self, dfs: list[pd.DataFrame], use_polars: bool = False
    ) -> tuple[pd.DataFrame, list[str], dict | None]:
        if use_polars:
            try:
                df_all_pl = pl.concat([pl.from_pandas(df) for df in dfs])
                df_all_pl = df_all_pl.replace([float("inf"), float("-inf")], None)
                df_all = df_all_pl.to_pandas()
            except (pl.exceptions.PolarsError, AttributeError, ImportError) as exc:  # pragma: no cover - optional dependency
                logger.exception("polars 未安装或初始化失败，回退至 pandas：%s", exc)
                df_all = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        else:
            df_all = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)

        base_cols_exist = [c for c in BASE_COLS if c in df_all.columns]
        other_cols = [c for c in df_all.columns if c not in base_cols_exist]
        df_all = df_all[base_cols_exist + other_cols]

        if not self.feature_cols_all:
            numeric_cols = [
                c
                for c in other_cols
                if (
                    pd.api.types.is_float_dtype(df_all[c])
                    or pd.api.types.is_integer_dtype(df_all[c])
                )
                and c != "bid_ask_imbalance"
            ]
            self.feature_cols_all = numeric_cols.copy()

        feat_cols_all = [
            c
            for c in self.feature_cols_all
            if c in df_all.columns
            and (
                pd.api.types.is_float_dtype(df_all[c])
                or pd.api.types.is_integer_dtype(df_all[c])
            )
            and c != "bid_ask_imbalance"
        ]
        feat_cols_all = [c for c in feat_cols_all if df_all[c].notna().any()]

        if self.mode == "train":
            scaler_params = {}
            scaled_parts = []
            for sym, g in df_all.groupby("symbol", group_keys=False):
                params = compute_robust_z_params(g, feat_cols_all)
                scaler_params[sym] = params
                scaled_parts.append(apply_robust_z_with_params(g, params))
            df_scaled = pd.concat(scaled_parts, ignore_index=True)
        else:
            if not self.scaler_path.is_file():
                raise FileNotFoundError(f"找不到 scaler 参数文件：{self.scaler_path}")
            scaler_params = load_scaler_params_from_json(str(self.scaler_path))
            scaled_parts = []
            for sym, g in df_all.groupby("symbol", group_keys=False):
                params = scaler_params.get(sym)
                if params is None:
                    params = compute_robust_z_params(g, feat_cols_all)
                scaled_parts.append(apply_robust_z_with_params(g, params))
            df_scaled = pd.concat(scaled_parts, ignore_index=True)

        df_scaled[feat_cols_all] = soft_clip(df_scaled[feat_cols_all], k=10.0)
        df_final, feat_cols_all = self._add_missing_flags(df_scaled, feat_cols_all)
        # 同步更新全局特征列表，确保后续批次字段一致
        self.feature_cols_all = feat_cols_all

        final_other_cols = [c for c in df_final.columns if c not in BASE_COLS]
        return df_final, final_other_cols, scaler_params if self.mode == "train" else None

    def _write_output(self, df: pd.DataFrame, save_to_db: bool, append: bool) -> None:
        self.merged_table_path.parent.mkdir(parents=True, exist_ok=True)
        if save_to_db:
            inspector = inspect(self.engine)
            table_exists = "features" in inspector.get_table_names()

            df = df.drop_duplicates(subset=["symbol", "interval", "open_time"])

            # MySQL 不支持直接插入 NaN，需要先替换为 None
            # 直接使用 where 会因列类型为 float 而无法保留 None，需先转为 object
            df = df.astype(object).where(pd.notna(df), None)

            if df.empty:
                logger.info("✅ 无新增 features 数据需要写入")
                return

            if not table_exists:
                df.to_sql("features", self.engine, if_exists="replace", index=False)
                logger.info("✅ 已创建并写入 MySQL 表 `features`")
            else:
                existing_cols = {c["name"] for c in inspector.get_columns("features")}
                missing = [c for c in df.columns if c not in existing_cols]
                dialect = self.engine.url.get_backend_name()
                for col in missing:
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "BIGINT"
                    elif (
                        pd.api.types.is_float_dtype(dtype)
                        or all(
                            (isinstance(v, (float, int)) or v is None)
                            for v in df[col]
                        )
                    ):
                        sql_type = "REAL" if dialect == "sqlite" else "DOUBLE"
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        sql_type = "DATETIME"
                    else:
                        sql_type = "TEXT"
                    with self.engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE features ADD COLUMN `{col}` {sql_type}"))
                cols = ", ".join(f"`{c}`" for c in df.columns)
                placeholders = ", ".join(f":{c}" for c in df.columns)
                if dialect == "sqlite":
                    sql = text(
                        f"INSERT OR IGNORE INTO features ({cols}) VALUES ({placeholders})"
                    )
                elif dialect.startswith("mysql"):
                    sql = text(
                        f"INSERT IGNORE INTO features ({cols}) VALUES ({placeholders})"
                    )
                else:
                    sql = text(f"INSERT INTO features ({cols}) VALUES ({placeholders})")
                records = []
                for rec in df.to_dict(orient="records"):
                    for k, v in rec.items():
                        if isinstance(v, pd.Timestamp):
                            rec[k] = v.to_pydatetime()
                    records.append(rec)
                try:
                    with self.engine.begin() as conn:
                        conn.execute(sql, records)
                    logger.info("✅ 已追加写入 MySQL 表 `features`")
                except IntegrityError as e:  # pragma: no cover - defensive
                    logger.warning("跳过部分重复行：%s", e)
        else:
            mode = "a" if append else "w"
            header = not append
            df.to_csv(self.merged_table_path, mode=mode, index=False, header=header)
            msg = "追加写入" if append else "导出"
            logger.info("✅ CSV 已%s → %s", msg, self.merged_table_path)

    def _calc_symbol_features(self, sym: str) -> Optional[pd.DataFrame]:
        """计算单个合约的所有特征并返回 DataFrame."""
        df_1h = self.load_klines_db(sym, "1h")
        df_4h = self.load_klines_db(sym, "4h")
        df_1d = self.load_klines_db(sym, "d1")
        try:
            df_5m = self.load_klines_db(sym, "5m")
        except (SQLAlchemyError, ValueError, KeyError) as exc:
            logger.exception("load_klines_db 5m failed for %s: %s", sym, exc)
            df_5m = None
        try:
            df_15m = self.load_klines_db(sym, "15m")
        except (SQLAlchemyError, ValueError, KeyError) as exc:
            logger.exception("load_klines_db 15m failed for %s: %s", sym, exc)
            df_15m = None

        if not all([df_1h is not None, df_4h is not None, df_1d is not None]):
            return None
        if len(df_1h) < 50 or len(df_4h) < 50 or len(df_1d) < 50:
            return None

        f1h = calc_features_raw(df_1h, "1h", symbol=sym)
        f4h = calc_features_raw(df_4h, "4h", symbol=sym)
        f1d = calc_features_raw(df_1d, "d1", symbol=sym)
        f5m = calc_features_raw(df_5m, "5m", symbol=sym) if df_5m is not None else None
        f15m = calc_features_raw(df_15m, "15m", symbol=sym) if df_15m is not None else None

        if f5m is not None:
            chg5 = f5m["pct_chg1_5m"].shift(1)
            f5m["mom_5m_roll1h"] = chg5.rolling(12, min_periods=1).mean()
            f5m["mom_5m_roll1h_std"] = chg5.rolling(12, min_periods=1).std()
            f5m = f5m[["mom_5m_roll1h", "mom_5m_roll1h_std"]]

        if f15m is not None:
            chg15 = f15m["pct_chg1_15m"].shift(1)
            f15m["mom_15m_roll1h"] = chg15.rolling(4, min_periods=1).mean()
            f15m["mom_15m_roll1h_std"] = chg15.rolling(4, min_periods=1).std()
            f15m = f15m[[
                "mom_15m_roll1h",
                "mom_15m_roll1h_std",
                "rsi_fast_15m",
                "stoch_fast_15m",
            ]]

        f1h = f1h.rename(columns={"close": "close_1h"})
        f4h = f4h.rename(columns={"close": "close_4h"})
        f1d = f1d.rename(columns={"close": "close_d1"})

        feats_all = [f1h, f4h, f1d]
        if f5m is not None:
            feats_all.append(f5m)
        if f15m is not None:
            feats_all.append(f15m)
        for feat in feats_all:
            feat.reset_index(inplace=True)
            feat.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")

        ob_df = self.load_order_book(sym)
        if self.include_order_book and ob_df is not None and len(ob_df) >= 50:
            ob_feat = calc_order_book_features(ob_df)
            ob_feat.reset_index(inplace=True)
            f1h = pd.merge_asof(
                f1h.sort_values("open_time"),
                ob_feat.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )

        merged = calc_cross_features_v2(f1h, f4h, f1d)
        if f5m is not None:
            merged = pd.merge_asof(
                merged.sort_values("open_time"),
                f5m.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        if f15m is not None:
            merged = pd.merge_asof(
                merged.sort_values("open_time"),
                f15m.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )

        raw = df_1h.reset_index()
        out = raw.merge(
            merged,
            on="open_time",
            how="left",
            suffixes=("", "_feat"),
        )

        out["symbol"] = sym
        out["hour_of_day"] = out["open_time"].dt.hour.astype(float)
        out["day_of_week"] = out["open_time"].dt.dayofweek.astype(float)
        out["hour_of_day_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
        out["hour_of_day_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
        out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
        if out["close"].nunique() > 1:
            out = self.add_up_down_targets(out, self.period_cfg)

        na_cols = out.columns[out.isna().all()]
        if len(na_cols):
            out.drop(columns=na_cols, inplace=True)

        raw_cols = list(self.cm_metrics)
        drop_cols = [c for c in raw_cols if c in out.columns]
        if drop_cols:
            out.drop(columns=drop_cols, inplace=True)

        # 保留所有链上衍生特征，不再因缺失率或数据量删除

        if out.shape[1] > 0:
            return out
        return None

    def merge_features(
        self,
        topn: int | None = None,
        symbols: Sequence[str] | None = None,
        save_to_db: bool = False,
        batch_size: int | None = None,
        n_jobs: int = 1,
        use_polars: bool = False,
    ) -> None:
        """合并多周期特征并写入文件或数据库。

        Parameters
        ----------
        topn
            仅处理成交量排名前 ``topn`` 的币种，默认为 ``self.topn``。
        symbols
            自定义币种列表，若为 ``None`` 则自动调用 :meth:`get_symbols`。
        save_to_db
            是否写入数据库。
        batch_size
            每处理 ``batch_size`` 个币种就落盘一次，``None`` 表示一次性处理。
        n_jobs
            并行作业数。
        use_polars
            是否使用 polars 加速合并。
        """

        if symbols is None:
            symbols = self.get_symbols(("1h", "4h", "d1", "5m", "15m"))
        symbols = list(symbols)[: (topn or self.topn)]

        all_dfs: list[pd.DataFrame] = []
        final_cols: set[str] = set()
        all_scaler_params: dict = {}
        append = False
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._calc_symbol_features)(sym)
                for sym in tqdm(symbols, desc="Calc features")
            )
        else:
            results = [
                self._calc_symbol_features(sym)
                for sym in tqdm(symbols, desc="Calc features")
            ]

        for out in results:
            if out is None:
                continue
            all_dfs.append(out)

            if batch_size and batch_size > 0 and len(all_dfs) >= batch_size:
                df_final, other_cols, scaler_params = self._finalize_batch(all_dfs, use_polars)
                self._write_output(df_final, save_to_db, append)
                final_cols.update(other_cols)
                if scaler_params:
                    all_scaler_params.update(scaler_params)
                all_dfs = []
                append = True

        if not all_dfs and not append:
            raise RuntimeError("合并结果为空——请确认数据库中三周期数据完整！")

        if batch_size and batch_size > 0 and all_dfs:
            df_final, other_cols, scaler_params = self._finalize_batch(all_dfs, use_polars)
            self._write_output(df_final, save_to_db, append)
            final_cols.update(other_cols)
            if scaler_params:
                all_scaler_params.update(scaler_params)
            all_dfs = []
            append = True
        elif not (batch_size and batch_size > 0):
            if use_polars:
                try:
                    df_all_pl = pl.concat([pl.from_pandas(df) for df in all_dfs])
                    df_all_pl = df_all_pl.replace([float("inf"), float("-inf")], None)
                    df_all = df_all_pl.to_pandas()
                except (pl.exceptions.PolarsError, AttributeError, ImportError) as exc:  # pragma: no cover - optional dependency
                    logger.exception("polars 未安装或初始化失败，回退至 pandas：%s", exc)
                    df_all = pd.concat(all_dfs, ignore_index=True).replace([
                        np.inf,
                        -np.inf,
                    ], np.nan)
            else:
                df_all = pd.concat(all_dfs, ignore_index=True).replace([
                    np.inf,
                    -np.inf,
                ], np.nan)
            df_final, other_cols, scaler_params = self._finalize_batch([df_all], use_polars)
            self._write_output(df_final, save_to_db, append=False)
            final_cols.update(other_cols)
            if scaler_params:
                all_scaler_params.update(scaler_params)
            all_dfs = []
            append = True

        if self.mode == "train":
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            save_scaler_params_to_json(all_scaler_params, str(self.scaler_path))
            if self.rise_transform == "boxcox":
                self.boxcox_lambda_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.boxcox_lambda_path, "w", encoding="utf-8") as f:
                    json.dump(self.boxcox_lambda, f)

        self.feature_cols_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_cols_path.write_text(
            "\n".join(sorted(final_cols)), encoding="utf-8"
        )
        logger.info("✅ feature_cols 保存至 %s", self.feature_cols_path)

    async def merge_features_async(
        self,
        topn: int | None = None,
        save_to_db: bool = False,
        batch_size: int | None = None,
        n_jobs: int = 1,
        use_polars: bool = False,
    ) -> None:
        """异步调用 :meth:`merge_features`"""
        await asyncio.to_thread(
            self.merge_features,
            topn=topn,
            save_to_db=save_to_db,
            batch_size=batch_size,
            n_jobs=n_jobs,
            use_polars=use_polars,
        )


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fe = FeatureEngineer("utils/config.yaml")
    fe.merge_features(save_to_db=True, batch_size=1)

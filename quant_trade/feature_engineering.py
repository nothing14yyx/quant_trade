# -*- coding: utf-8 -*-
"""
FeatureEngineer v2.3-patch1 (External indicators removed)  (2025-06-03)
==================================================================
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from sqlalchemy import create_engine, text
from sklearn.metrics import mutual_info_score

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

        self.period_cfg: dict[str, dict[str, float | int]] = {
            "1h": {"q_low": 0.25, "q_up": 0.75, "base_n": 3, "vol_window": 24},
            "4h": {"q_low": 0.30, "q_up": 0.70, "base_n": 12, "vol_window": 96},
            "d1": {"q_low": 0.35, "q_up": 0.65, "base_n": 72, "vol_window": 576},
        }

        self._kl_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        # 保存/加载剪裁+缩放参数的 JSON 路径
        self.scaler_path: Path = Path(
            fe_cfg.get("scaler_path", "scalers/all_features_scaler.json")
        )

        # 从 feature_cols.txt 自动读取所有待标准化列；若文件不存在，先设为空列表
        if self.feature_cols_path.is_file():
            lines = self.feature_cols_path.read_text(encoding="utf-8").splitlines()
            self.feature_cols_all: List[str] = [c.strip() for c in lines if c.strip()]
        else:
            self.feature_cols_all = []

    @staticmethod
    def add_up_down_targets(
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

                up_thr = up_ret.quantile(q_up)
                down_thr = down_ret.quantile(q_low)

                # allow both thresholds to be on the same side of zero when the
                # market shows a persistent trend. Only ensure `down_thr` is
                # strictly smaller than `up_thr` to avoid degenerate targets.
                if not (down_thr < up_thr):
                    raise ValueError(
                        f"Invalid thresholds for {p}: {down_thr}, {up_thr}"
                    )

                if classification_mode != "three":
                    raise ValueError("Only 'three' classification_mode is supported")

                target = np.where(up_ret >= up_thr, 2, np.where(down_ret <= down_thr, 0, 1))

                g[f"target_{p}"] = target.astype(float)
                g[f"future_volatility_{p}"] = close.pct_change().rolling(base_n).std().shift(-base_n)
                g[f"future_max_rise_{p}"] = up_ret
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
        return list(set.intersection(*symbol_sets))

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
            q = (
                "SELECT timestamp, metric, value FROM cm_onchain_metrics "
                "WHERE symbol=:sym AND metric IN "
                "('AdrActCnt','AdrNewCnt','TxCnt','CapMrktCurUSD','CapRealUSD') "
                "ORDER BY timestamp"
            )
            cm_df = pd.read_sql(text(q), self.engine, params={"sym": symbol}, parse_dates=["timestamp"])
        except Exception:  # pragma: no cover - optional table
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
            out["AdrActCnt"] = None
            out["AdrNewCnt"] = None
            out["TxCnt"] = None
            out["CapMrktCurUSD"] = None
            out["CapRealUSD"] = None

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

        # ----- 删除缺失率 ≥95% 的列 -----
        drop_cols = missing_ratio[missing_ratio >= 0.95].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            feat_cols = [c for c in feat_cols if c not in drop_cols]
            missing_ratio = missing_ratio.drop(drop_cols)

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
            except Exception as e:  # pragma: no cover - optional dependency
                logger.warning("polars 未安装或初始化失败，回退至 pandas：%s", e)
                df_all = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        else:
            df_all = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)

        base_cols = [
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
        # FUTURE_COLS 中包含 "target"，拼接后需去重，避免生成重复列
        base_cols = list(dict.fromkeys(base_cols))
        base_cols_exist = [c for c in base_cols if c in df_all.columns]
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

        final_other_cols = [c for c in df_final.columns if c not in base_cols]
        return df_final, final_other_cols, scaler_params if self.mode == "train" else None

    def _write_output(self, df: pd.DataFrame, save_to_db: bool, append: bool) -> None:
        self.merged_table_path.parent.mkdir(parents=True, exist_ok=True)
        if save_to_db:
            if_exists = "append" if append else "replace"
            df.to_sql("features", self.engine, if_exists=if_exists, index=False)
            msg = "追加写入" if append else "写入"
            logger.info("✅ 已%s MySQL 表 `features`", msg)
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
        except Exception:
            df_5m = None
        try:
            df_15m = self.load_klines_db(sym, "15m")
        except Exception:
            df_15m = None

        if not all([df_1h is not None, df_4h is not None, df_1d is not None]):
            return None
        if len(df_1h) < 50 or len(df_4h) < 50 or len(df_1d) < 50:
            return None

        f1h = calc_features_raw(df_1h, "1h")
        f4h = calc_features_raw(df_4h, "4h")
        f1d = calc_features_raw(df_1d, "d1")
        f5m = calc_features_raw(df_5m, "5m") if df_5m is not None else None
        f15m = calc_features_raw(df_15m, "15m") if df_15m is not None else None

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

        merged = calc_cross_features(f1h, f4h, f1d)
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
        if out["close"].nunique() > 1:
            out = self.add_up_down_targets(out, self.period_cfg)

        na_cols = out.columns[out.isna().all()]
        if len(na_cols):
            out.drop(columns=na_cols, inplace=True)

        if out.shape[1] > 0:
            return out
        return None

    def merge_features(
        self,
        topn: int | None = None,
        save_to_db: bool = False,
        batch_size: int | None = None,
        n_jobs: int = 1,
        use_polars: bool = False,
    ) -> None:
        symbols = self.get_symbols(("1h", "4h", "d1", "5m", "15m"))
        symbols = symbols[: (topn or self.topn)]

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
                except Exception as e:  # pragma: no cover - optional dependency
                    logger.warning("polars 未安装或初始化失败，回退至 pandas：%s", e)
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

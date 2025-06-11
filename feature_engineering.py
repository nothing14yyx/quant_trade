# -*- coding: utf-8 -*-
"""
FeatureEngineer v2.3-patch1 (External indicators removed)  (2025-06-03)
==================================================================
在原版 FeatureEngineer 的基础上，去掉所有“外部指标”相关的逻辑：
  - 不再假设 klines 表里存在 vix、dxy、btc_dominance、adractcnt_*、txcnt_*、feetotntv_* 等列。
  - merge_features 中的 DROP_PREFIXES 不再包含这些外部前缀。
  - 其余特征计算依赖于 utils/helper.calc_features_full，它也已移除了外部字段。
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sqlalchemy import create_engine

# 不再 import calc_features_full，而改为：
from utils.helper import calc_features_raw, calc_order_book_features  # pylint: disable=import-error

# Robust-z 参数持久化工具
from utils.robust_scaler import (
    compute_robust_z_params,
    save_scaler_params_to_json,
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)
from utils.feature_health import health_check, apply_health_check_df

# Future-related columns to drop for leakage prevention
FUTURE_COLS = [
    "future_volatility",
    "future_max_rise",
    "future_max_drawdown",
]

class FeatureEngineer:
    """多周期特征工程生成器 (1h → 4h → 1d)。

    主要修复 / 改进
    ----------------
    1. 无未来数据泄漏的 add_up_down_targets。
    2. 从 feature_cols.txt 动态读取所有待标准化特征，无需手写列表。
    3. 支持 Robust-z 参数持久化：训练时算出 p1/p99/mean/std 并保存；推理时直接加载。
    4. 过滤掉非数值列，只对数值列做 Robust-z。
    5. _add_missing_flags 中一次性 concat，避免 DataFrame 碎片化。
    6. 去掉外部指标（vix/dxy/链上等）相关逻辑，无需再清理这些列。
    """

    def __init__(self, config_path: str | os.PathLike = "utils/config.yaml") -> None:
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
        self.feature_cols_path: Path = Path(fe_cfg.get("feature_cols_path", "data/merged/feature_cols.txt"))
        self.merged_table_path: Path = Path(fe_cfg.get("merged_table_path", "data/merged/merged_table.csv"))

        # Robust-z 模式：train 或 inference
        self.mode: str = fe_cfg.get("mode", "train")
        # 保存/加载剪裁+缩放参数的 JSON 路径
        self.scaler_path: Path = Path(fe_cfg.get("scaler_path", "scalers/all_features_scaler.json"))

        # 从 feature_cols.txt 自动读取所有待标准化列；若文件不存在，先设为空列表
        if self.feature_cols_path.is_file():
            lines = self.feature_cols_path.read_text(encoding="utf-8").splitlines()
            self.feature_cols_all: List[str] = [c.strip() for c in lines if c.strip()]
        else:
            self.feature_cols_all = []

    @staticmethod
    def add_up_down_targets(
        df: pd.DataFrame,
        threshold: float | str | None = "balanced",
        shift_n: int | str = "dynamic",
        vol_window: int = 24,
        n_bins: int | None = None,
    ) -> pd.DataFrame:
        """生成涨跌与波动率等标签，无未来数据泄漏

        参数 threshold 支持：
            - float：固定阈值
            - "auto": 使用 rolling mean 波动率 * 1.5
            - "quantile": 使用 rolling 80% 分位波动率
            - "balanced": 使用全局分位并控制正样本在 45%~55%
            - None：等同于 "auto"
        """

        results = []
        atr_col = next((c for c in df.columns if c.startswith("atr_pct")), None)
        for sym, g in df.groupby("symbol", group_keys=False):
            if shift_n == "dynamic" and atr_col is not None:
                mean_atr = g[atr_col].mean()
                n = 2 if mean_atr > 0.07 else 4
            elif isinstance(shift_n, int):
                n = shift_n
            else:
                n = 3

            close = g["close"]
            fut_hi = close.shift(-1).rolling(n, min_periods=1).max()
            fut_lo = close.shift(-1).rolling(n, min_periods=1).min()

            if threshold in (None, "auto"):
                vol = (
                    close.pct_change()
                    .abs()
                    .rolling(vol_window, min_periods=1)
                    .mean()
                ) * 1.5
                th_up = th_down = vol
            elif threshold == "quantile":
                th = (
                    close.pct_change()
                    .abs()
                    .rolling(vol_window, min_periods=1)
                    .quantile(0.8)
                )
                th_up = th_down = th
            elif threshold == "balanced":
                chg_up = fut_hi / close - 1
                chg_down = fut_lo / close - 1
                th_up = pd.Series(chg_up.quantile(0.55), index=g.index)
                th_down = pd.Series(chg_down.quantile(0.45), index=g.index)
            else:
                th_up = th_down = pd.Series(float(threshold), index=g.index)

            up_ret = fut_hi / close - 1
            down_ret = fut_lo / close - 1
            g["target_up"] = (up_ret >= th_up).astype(float)
            g["target_down"] = (down_ret <= th_down).astype(float)
            if n_bins and n_bins > 1:
                bins_up = np.quantile(up_ret.dropna(), np.linspace(0, 1, n_bins + 1))
                bins_down = np.quantile(down_ret.dropna(), np.linspace(0, 1, n_bins + 1))
                g["target_up_multi"] = pd.cut(up_ret, bins=bins_up, labels=False, include_lowest=True)
                g["target_down_multi"] = pd.cut(down_ret, bins=bins_down, labels=False, include_lowest=True)
            g["future_volatility"] = (
                close.pct_change()
                .rolling(n)
                .std()
                .shift(-n)
            )
            g["future_max_rise"] = fut_hi / close - 1
            g["future_max_drawdown"] = fut_lo / close - 1
            drop_cols = ["target_up", "target_down", "future_volatility"]
            if n_bins and n_bins > 1:
                drop_cols += ["target_up_multi", "target_down_multi"]
            g.loc[g.tail(n).index, drop_cols] = np.nan
            results.append(g)

        return pd.concat(results).sort_index()

    def get_symbols(self, intervals: tuple[str, str, str] = ("1h", "4h", "1d")) -> List[str]:
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

    def load_klines_db(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """读取指定 symbol/interval 的 K 线，若无数据返回 None。"""
        sql = "SELECT * FROM klines WHERE symbol=%s AND `interval`=%s ORDER BY open_time"
        df = pd.read_sql(sql, self.engine, parse_dates=["open_time", "close_time"], params=(symbol, interval))
        if df.empty:
            return None
        return df.set_index("open_time").sort_index()

    def load_order_book(self, symbol: str) -> Optional[pd.DataFrame]:
        """读取指定合约的 order_book 快照"""
        sql = "SELECT * FROM order_book WHERE symbol=%s ORDER BY timestamp"
        df = pd.read_sql(sql, self.engine, params=(symbol,), parse_dates=["timestamp"])
        if df.empty:
            return None
        return df

    def _add_missing_flags(self, df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
        """Forward fill within each symbol and optionally add isna flags."""

        missing_ratio = df[feat_cols].isna().mean()
        flag_cols = [c for c in feat_cols if 0 < missing_ratio[c] < 0.95]

        flags_df = df[flag_cols].isna().astype(int)
        flags_df.columns = [f"{col}_isnan" for col in flag_cols]

        df_filled = df.copy()
        if "symbol" in df_filled.columns:
            df_filled[feat_cols] = (
                df_filled.groupby("symbol", group_keys=False)[feat_cols].ffill()
            )
        else:
            df_filled[feat_cols] = df_filled[feat_cols].ffill()
        df_filled[feat_cols] = df_filled[feat_cols].fillna(0.0)

        df_out = pd.concat([df_filled, flags_df], axis=1)
        return df_out

    def _finalize_batch(self, dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
        df_all = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)

        base_cols = [
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
            "symbol", "target_up", "target_down",
        ] + FUTURE_COLS
        df_all.drop(columns=[c for c in FUTURE_COLS if c in df_all.columns], inplace=True)
        other_cols = [c for c in df_all.columns if c not in base_cols]
        df_all = df_all[base_cols + other_cols]

        if not self.feature_cols_all:
            numeric_cols = [
                c for c in other_cols
                if pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            ]
            self.feature_cols_all = numeric_cols.copy()

        feat_cols_all = [
            c for c in self.feature_cols_all
            if c in df_all.columns and (
                pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            )
        ]


        if self.mode == "train":
            scaler_params = compute_robust_z_params(df_all, feat_cols_all)
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            save_scaler_params_to_json(scaler_params, str(self.scaler_path))
            df_scaled = apply_robust_z_with_params(df_all, scaler_params)
        else:
            if not self.scaler_path.is_file():
                raise FileNotFoundError(f"找不到 scaler 参数文件：{self.scaler_path}")
            scaler_params = load_scaler_params_from_json(str(self.scaler_path))
            df_scaled = apply_robust_z_with_params(df_all, scaler_params)

        df_scaled[feat_cols_all] = df_scaled[feat_cols_all].clip(-30, 30)
        df_final = self._add_missing_flags(df_scaled, feat_cols_all)
        df_final.drop(columns=[c for c in FUTURE_COLS if c in df_final.columns], inplace=True)

        final_other_cols = [c for c in df_final.columns if c not in base_cols]
        return df_final, final_other_cols

    def _write_output(self, df: pd.DataFrame, save_to_db: bool, append: bool) -> None:
        self.merged_table_path.parent.mkdir(parents=True, exist_ok=True)
        if save_to_db:
            if_exists = "append" if append else "replace"
            df.to_sql("features", self.engine, if_exists=if_exists, index=False)
            msg = "追加写入" if append else "写入"
            print(f"✅ 已{msg} MySQL 表 `features`")
        else:
            mode = "a" if append else "w"
            header = not append
            df.to_csv(self.merged_table_path, mode=mode, index=False, header=header)
            msg = "追加写入" if append else "导出"
            print(f"✅ CSV 已{msg} → {self.merged_table_path}")

    def merge_features(
        self,
        topn: int | None = None,
        save_to_db: bool = False,
        batch_size: int | None = None,
    ) -> None:
        symbols = self.get_symbols(("1h", "4h", "1d", "5m", "15m"))
        symbols = symbols[: (topn or self.topn)]

        all_dfs: list[pd.DataFrame] = []
        final_cols: set[str] = set()
        append = False
        for sym in tqdm(symbols, desc="Calc features"):
            # 1. 先拉取各周期数据
            df_1h = self.load_klines_db(sym, "1h")
            df_4h = self.load_klines_db(sym, "4h")
            df_1d = self.load_klines_db(sym, "1d")
            try:
                df_5m = self.load_klines_db(sym, "5m")
            except Exception:
                df_5m = None
            try:
                df_15m = self.load_klines_db(sym, "15m")
            except Exception:
                df_15m = None
            if not all([df_1h is not None, df_4h is not None, df_1d is not None]):
                continue

            # 2. 如果某个周期数据不足以计算所有指标（ema50 需要至少 50 条），则跳过
            if len(df_1h) < 50 or len(df_4h) < 50 or len(df_1d) < 50:
                continue

            # 3. 计算各周期“原始”特征（不做剪裁/归一化）
            f1h = calc_features_raw(df_1h, "1h")
            f4h = calc_features_raw(df_4h, "4h")
            f1d = calc_features_raw(df_1d, "d1")
            f5m = calc_features_raw(df_5m, "5m") if df_5m is not None else None
            f15m = calc_features_raw(df_15m, "15m") if df_15m is not None else None

            if f5m is not None:
                f5m["mom_5m_roll1h"] = f5m["pct_chg1_5m"].rolling(12, min_periods=1).mean()
                f5m["mom_5m_roll1h_std"] = f5m["pct_chg1_5m"].rolling(12, min_periods=1).std()
                f5m = f5m[["mom_5m_roll1h", "mom_5m_roll1h_std"]]

            if f15m is not None:
                f15m["mom_15m_roll1h"] = f15m["pct_chg1_15m"].rolling(4, min_periods=1).mean()
                f15m["mom_15m_roll1h_std"] = f15m["pct_chg1_15m"].rolling(4, min_periods=1).std()
                f15m = f15m[["mom_15m_roll1h", "mom_15m_roll1h_std"]]

            # 4. 将各周期特征表中的 close 重命名，避免与 raw.close 冲突
            f1h = f1h.rename(columns={"close": "close_1h"})
            f4h = f4h.rename(columns={"close": "close_4h"})
            f1d = f1d.rename(columns={"close": "close_d1"})

            # 5. 重命名索引列“index”为“open_time”，以便 merge_asof
            feats_all = [f1h, f4h, f1d]
            if f5m is not None:
                feats_all.append(f5m)
            if f15m is not None:
                feats_all.append(f15m)
            for feat in feats_all:
                feat.reset_index(inplace=True)
                feat.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")

            ob_df = self.load_order_book(sym)
            if ob_df is not None and not ob_df.empty:
                ob_feat = calc_order_book_features(ob_df)
                ob_feat.reset_index(inplace=True)
                f1h = pd.merge_asof(
                    f1h.sort_values("open_time"),
                    ob_feat.sort_values("open_time"),
                    on="open_time",
                    direction="backward",
                )

            # ===== 修改开始：计算真实收盘时间并删掉多余 open_time =====
            f4h["close_time_4h"] = f4h["open_time"] + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
            f1d["close_time_d1"] = f1d["open_time"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            f4h = f4h.drop(columns=["open_time"])
            f1d = f1d.drop(columns=["open_time"])
            # ===== 修改结束 =====

            # 6. merge_asof 对齐 1h→4h→1d：务必对齐到已经收盘的那根 K 线
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

            # --- 跨周期衍生特征 ---
            merged["close_spread_1h_4h"] = merged["close_1h"] - merged["close_4h"]
            merged["close_spread_1h_d1"] = merged["close_1h"] - merged["close_d1"]
            merged["ma_ratio_1h_4h"] = merged["sma_10_1h"] / merged["sma_10_4h"].replace(0, np.nan)
            merged["ma_ratio_1h_d1"] = merged["sma_10_1h"] / merged["sma_10_d1"].replace(0, np.nan)
            merged["ma_ratio_4h_d1"] = merged["sma_10_4h"] / merged["sma_10_d1"].replace(0, np.nan)
            merged["atr_pct_ratio_1h_4h"] = merged["atr_pct_1h"] / merged["atr_pct_4h"].replace(0, np.nan)
            merged["bb_width_ratio_1h_4h"] = merged["bb_width_1h"] / merged["bb_width_4h"].replace(0, np.nan)
            merged["rsi_diff_1h_4h"] = merged["rsi_1h"] - merged["rsi_4h"]
            merged["rsi_diff_1h_d1"] = merged["rsi_1h"] - merged["rsi_d1"]
            merged["rsi_diff_4h_d1"] = merged["rsi_4h"] - merged["rsi_d1"]
            merged["macd_hist_diff_1h_4h"] = merged["macd_hist_1h"] - merged["macd_hist_4h"]
            merged["macd_hist_diff_1h_d1"] = merged["macd_hist_1h"] - merged["macd_hist_d1"]
            merged["macd_hist_4h_mul_bb_width_1h"] = merged["macd_hist_4h"] * merged["bb_width_1h"]
            merged["rsi_1h_mul_vol_ma_ratio_4h"] = merged["rsi_1h"] * merged["vol_ma_ratio_4h"]
            merged["macd_hist_1h_mul_bb_width_4h"] = merged["macd_hist_1h"] * merged["bb_width_4h"]
            merged["vol_ratio_1h_4h"] = merged["vol_ma_ratio_1h"] / merged["vol_ma_ratio_4h"].replace(0, np.nan)
            merged["vol_ratio_4h_d1"] = merged["vol_ma_ratio_4h"] / merged["vol_ma_ratio_d1"].replace(0, np.nan)

            # 7. 将原始 1h K 线回拼回 merged，打上 target_up/target_down
            raw = df_1h.reset_index()  # raw["open_time"] 是 datetime64[ns]

            out = raw.merge(
                merged,
                on="open_time",
                how="left",
                suffixes=("", "_feat")
            )

            out["symbol"] = sym
            out["hour_of_day"] = out["open_time"].dt.hour.astype(float)
            out["day_of_week"] = out["open_time"].dt.dayofweek.astype(float)
            out = self.add_up_down_targets(out)

            # 8. 删除完全为 NaN 的列
            na_cols = out.columns[out.isna().all()]
            if len(na_cols):
                out.drop(columns=na_cols, inplace=True)

            if out.shape[1] > 0:
                all_dfs.append(out)

            if batch_size and batch_size > 0 and len(all_dfs) >= batch_size:
                df_final, other_cols = self._finalize_batch(all_dfs)
                self._write_output(df_final, save_to_db, append)
                final_cols.update(other_cols)
                all_dfs = []
                append = True

        if not all_dfs and not append:
            raise RuntimeError("合并结果为空——请确认数据库中三周期数据完整！")

        if batch_size and batch_size > 0 and all_dfs:
            df_final, other_cols = self._finalize_batch(all_dfs)
            self._write_output(df_final, save_to_db, append)
            final_cols.update(other_cols)
            all_dfs = []
            append = True
        elif not (batch_size and batch_size > 0):
            df_all = pd.concat(all_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            df_final, other_cols = self._finalize_batch([df_all])
            self._write_output(df_final, save_to_db, append=False)
            final_cols.update(other_cols)
            all_dfs = []
            append = True

        self.feature_cols_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_cols_path.write_text("\n".join(sorted(final_cols)), encoding="utf-8")
        print(f"✅ feature_cols 保存至 {self.feature_cols_path}")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fe = FeatureEngineer("utils/config.yaml")
    fe.merge_features(save_to_db=True, batch_size=1)

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
from utils.helper import calc_features_raw  # pylint: disable=import-error

# Robust-z 参数持久化工具
from utils.robust_scaler import (
    compute_robust_z_params,
    save_scaler_params_to_json,
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)
from utils.feature_health import health_check, apply_health_check_df


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
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        # MySQL 连接配置
        mysql_cfg = self.cfg.get("mysql", {})
        self.engine = create_engine(
            f"mysql+pymysql://{mysql_cfg.get('user', 'root')}:{mysql_cfg.get('password', '')}@"
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
    def add_up_down_targets(df: pd.DataFrame, threshold: float = 0.015, shift_n: int = 3) -> pd.DataFrame:
        """向量化生成 target_up / target_down，避免 Python for-loop。（无未来泄漏）"""
        close = df["close"]
        fut_hi = close.shift(-1).rolling(shift_n, min_periods=1).max()
        fut_lo = close.shift(-1).rolling(shift_n, min_periods=1).min()
        df["target_up"] = ((fut_hi / close - 1) >= threshold).astype(float)
        df["target_down"] = ((fut_lo / close - 1) <= -threshold).astype(float)
        df.loc[df.tail(shift_n).index, ["target_up", "target_down"]] = np.nan
        return df

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

    def _add_missing_flags(self, df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
        # 一次性生成所有 isna 标志列，消除碎片 warning
        flags_df = df[feat_cols].isna().astype(int)
        flags_df.columns = [f"{col}_isnan" for col in feat_cols]
        df_filled = df.copy()
        df_filled[feat_cols] = df_filled[feat_cols].fillna(0.0)
        df_out = pd.concat([df_filled, flags_df], axis=1)
        return df_out

    def merge_features(self, topn: int | None = None, save_to_db: bool = False) -> None:
        symbols = self.get_symbols()
        symbols = symbols[: (topn or self.topn)]

        all_dfs: list[pd.DataFrame] = []
        for sym in tqdm(symbols, desc="Calc features"):
            # 1. 先拉取 1h/4h/1d 数据
            df_1h = self.load_klines_db(sym, "1h")
            df_4h = self.load_klines_db(sym, "4h")
            df_1d = self.load_klines_db(sym, "1d")
            if not all([df_1h is not None, df_4h is not None, df_1d is not None]):
                continue

            # 2. 如果某个周期数据不足以计算所有指标（ema50 需要至少 50 条），则跳过
            if len(df_1h) < 50 or len(df_4h) < 50 or len(df_1d) < 50:
                continue

            # 3. 计算各周期“原始”特征（不做剪裁/归一化）
            f1h = calc_features_raw(df_1h, "1h")
            f4h = calc_features_raw(df_4h, "4h")
            f1d = calc_features_raw(df_1d, "d1")

            # 4. 将各周期特征表中的 close 重命名，避免与 raw.close 冲突
            f1h = f1h.rename(columns={"close": "close_1h"})
            f4h = f4h.rename(columns={"close": "close_4h"})
            f1d = f1d.rename(columns={"close": "close_d1"})

            # 5. 重命名索引列“index”为“open_time”，以便 merge_asof
            for feat in (f1h, f4h, f1d):
                feat.reset_index(inplace=True)
                feat.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")

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

            # --- 跨周期衍生特征 ---
            merged["close_spread_1h_4h"] = merged["close_1h"] - merged["close_4h"]
            merged["close_spread_1h_d1"] = merged["close_1h"] - merged["close_d1"]
            merged["ma_ratio_1h_4h"] = merged["sma_10_1h"] / merged["sma_10_4h"].replace(0, np.nan)
            merged["ma_ratio_1h_d1"] = merged["sma_10_1h"] / merged["sma_10_d1"].replace(0, np.nan)
            merged["atr_pct_ratio_1h_4h"] = merged["atr_pct_1h"] / merged["atr_pct_4h"].replace(0, np.nan)
            merged["bb_width_ratio_1h_4h"] = merged["bb_width_1h"] / merged["bb_width_4h"].replace(0, np.nan)

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

        if not all_dfs:
            raise RuntimeError("合并结果为空——请确认数据库中三周期数据完整！")

        # 9. 拼成一张总表
        df_all = pd.concat(all_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)

        # 10. 基础列顺序固定，其他特征列放后面
        base_cols = [
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
            "symbol", "target_up", "target_down",
        ]
        other_cols = [c for c in df_all.columns if c not in base_cols]
        df_all = df_all[base_cols + other_cols]

        # 11. 如果第一次没有 feature_cols_all，就把 other_cols 中的数值列设为初始待归一化列表
        if not self.feature_cols_all:
            numeric_cols = [
                c for c in other_cols
                if pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            ]
            self.feature_cols_all = numeric_cols.copy()

        # 12. 过滤 feature_cols_all，确保只保留在 df_all 中存在且为数值列的字段
        feat_cols_all = [
            c for c in self.feature_cols_all
            if c in df_all.columns and (
                pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            )
        ]

        # ===== 第一步：对 df_all 中每个特征按 1%/99% 分位数裁剪 =====
        for col in feat_cols_all:
            arr = df_all[col].dropna().values
            low, high = np.nanpercentile(arr, [1, 99])
            df_all[col] = df_all[col].clip(low, high)

        # ===== 第二步：Robust-z 缩放（训练 vs 推理模式） =====
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

        # ===== 第三步：对归一化后的结果统一 Clip 到 [-10, 10] =====
        df_scaled[feat_cols_all] = df_scaled[feat_cols_all].clip(-10, 10)

        # 13. 添加 _isnan 标志并把 NaN 填为 0
        df_final = self._add_missing_flags(df_scaled, feat_cols_all)

        # 14. 最终输出：写入 MySQL 或导出 CSV
        self.merged_table_path.parent.mkdir(parents=True, exist_ok=True)
        if save_to_db:
            df_final.to_sql("features", self.engine, if_exists="replace", index=False)
            print("✅ 已写入 MySQL 表 `features`")
        else:
            df_final.to_csv(self.merged_table_path, index=False)
            print(f"✅ CSV 已导出 → {self.merged_table_path}")

        # 15. 更新 feature_cols.txt（除去 base_cols）
        final_other_cols = [c for c in df_final.columns if c not in base_cols]
        self.feature_cols_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_cols_path.write_text("\n".join(final_other_cols), encoding="utf-8")
        print(f"✅ feature_cols 保存至 {self.feature_cols_path}")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fe = FeatureEngineer("utils/config.yaml")
    fe.merge_features(save_to_db=True)

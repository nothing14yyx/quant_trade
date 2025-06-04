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
from sqlalchemy import create_engine
from tqdm import tqdm

# calc_features_full 已在 helper 里移除了外部指标
from utils.helper import calc_features_full  # pylint: disable=import-error

# Robust-z 参数持久化工具
from utils.robust_scaler import (
    compute_robust_z_params,
    save_scaler_params_to_json,
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)


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

    def _add_missing_flags(self, df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
        """
        为每个特征列批量创建 _isnan 标志，并把缺失值填 0。
        为避免 DataFrame 碎片化，先在一个新的 DataFrame 中生成所有 flag，再 concat。
        """
        # 1. 构造一个空的 flags DataFrame，行索引与原 df 相同
        flags_df = pd.DataFrame(index=df.index)
        for col in feat_cols:
            flag_col = f"{col}_isnan"
            # 1) 先把标志放入 flags_df
            flags_df[flag_col] = df[col].isna().astype(int)

        # 2. 将 flags_df 中的所有列填 0 再合并到原 df 之前的数值列
        df_filled = df.copy()
        df_filled[feat_cols] = df_filled[feat_cols].fillna(0.0)

        # 3. 一次性合并 flags_df
        df_out = pd.concat([df_filled, flags_df], axis=1)
        return df_out

    def merge_features(self, topn: int | None = None, save_to_db: bool = False) -> None:
        symbols = self.get_symbols()
        symbols = symbols[: (topn or self.topn)]

        all_dfs: list[pd.DataFrame] = []
        for sym in tqdm(symbols, desc="Calc features"):
            df_1h = self.load_klines_db(sym, "1h")
            df_4h = self.load_klines_db(sym, "4h")
            df_1d = self.load_klines_db(sym, "1d")
            if not all([df_1h is not None, df_4h is not None, df_1d is not None]):
                continue

            # 1. 计算多周期特征（calc_features_full 已不再产生外部指标列）
            f1h = calc_features_full(df_1h, "1h")
            f4h = calc_features_full(df_4h, "4h")
            f1d = calc_features_full(df_1d, "d1")

            for feat in (f1h, f4h, f1d):
                feat.reset_index(inplace=True)
                feat.rename(columns={"index": "open_time"}, inplace=True, errors="ignore")

            # 2. merge_asof 对齐
            merged = pd.merge_asof(
                f1h.sort_values("open_time"),
                f4h.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
            merged = pd.merge_asof(
                merged.sort_values("open_time"),
                f1d.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )

            # 3. 拼回原 1h K 线 & 打标签
            raw = df_1h.reset_index()
            out = raw.merge(merged, on="open_time", how="left")
            out["symbol"] = sym
            out = self.add_up_down_targets(out)

            # 4. 清理全 NaN 列，避免后续 concat 出现问题
            na_cols = out.columns[out.isna().all()]
            if len(na_cols):
                out.drop(columns=na_cols, inplace=True)

            if out.shape[1] > 0:
                all_dfs.append(out)

        if not all_dfs:
            raise RuntimeError("合并结果为空——请确认数据库中三周期数据完整！")

        df_all = pd.concat(all_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan)

        base_cols = [
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
            "symbol", "target_up", "target_down",
        ]
        other_cols = [c for c in df_all.columns if c not in base_cols]
        df_all = df_all[base_cols + other_cols]

        # 如果是第一次生成 feature_cols.txt，这时 feature_cols_all 为空，则先把 other_cols 中的数值列赋给它
        if not self.feature_cols_all:
            numeric_cols = [
                c for c in other_cols
                if pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            ]
            self.feature_cols_all = numeric_cols.copy()

        # 过滤 feat_cols_all，只保留在 df_all 中存在且为数值列的部分
        feat_cols_all = [
            c for c in self.feature_cols_all
            if c in df_all.columns and (
                pd.api.types.is_float_dtype(df_all[c]) or pd.api.types.is_integer_dtype(df_all[c])
            )
        ]

        # Robust-z：训练时计算+保存参数，推理时加载+应用
        if self.mode == "train":
            scaler_params = compute_robust_z_params(df_all, feat_cols_all)
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            save_scaler_params_to_json(scaler_params, str(self.scaler_path))
            df_scaled = apply_robust_z_with_params(df_all, scaler_params)
        else:  # inference
            if not self.scaler_path.is_file():
                raise FileNotFoundError(f"找不到 scaler 参数文件：{self.scaler_path}")
            scaler_params = load_scaler_params_from_json(str(self.scaler_path))
            df_scaled = apply_robust_z_with_params(df_all, scaler_params)

        # 添加 _isnan 标志并填充 0（使用一次性 concat，避免碎片化）
        df_final = self._add_missing_flags(df_scaled, feat_cols_all)

        # 输出到 CSV 或写入 MySQL
        self.merged_table_path.parent.mkdir(parents=True, exist_ok=True)
        if save_to_db:
            df_final.to_sql("features", self.engine, if_exists="replace", index=False)
            print("✅ 已写入 MySQL 表 `features`")
        else:
            df_final.to_csv(self.merged_table_path, index=False)
            print(f"✅ CSV 已导出 → {self.merged_table_path}")

        # 保存特征列名（除了 base_cols 之外），写回 feature_cols.txt
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

from __future__ import annotations

from collections import deque
import pandas as pd
import numpy as np


class FeatureProcessor:
    """处理输入特征并提供常用逻辑"""

    def __init__(self):
        self._std_index_cache: dict[str, tuple | None] = {
            p: None for p in ("15m", "1h", "4h", "d1")
        }

    def normalize_features(self, feats, period: str) -> dict:
        """将 DataFrame/Series 输入转为字典并缓存索引"""
        if isinstance(feats, dict):
            return feats
        if isinstance(feats, pd.Series):
            cols = tuple(feats.index)
            cache = self._std_index_cache.get(period)
            if cache is None or cache[0] != cols:
                idx_map = {c: i for i, c in enumerate(cols)}
                self._std_index_cache[period] = (cols, idx_map)
            else:
                idx_map = cache[1]
            return {c: feats.iat[i] for c, i in idx_map.items()}
        if isinstance(feats, pd.DataFrame) and not feats.empty:
            row = feats.iloc[-1]
            cols = tuple(row.index)
            cache = self._std_index_cache.get(period)
            if cache is None or cache[0] != cols:
                idx_map = {c: i for i, c in enumerate(cols)}
                self._std_index_cache[period] = (cols, idx_map)
            else:
                idx_map = cache[1]
            return {c: row.iat[i] for c, i in idx_map.items()}
        return {}

    @staticmethod
    def ma_cross_logic(features: dict, sma_20_1h_prev=None) -> float:
        sma5 = features.get("sma_5_1h")
        sma20 = features.get("sma_20_1h")
        ma_ratio = features.get("ma_ratio_5_20", 1.0)
        if sma5 is None or sma20 is None:
            return 1.0
        slope = 0.0
        if sma_20_1h_prev not in (None, 0):
            slope = (sma20 - sma_20_1h_prev) / sma_20_1h_prev
        if (ma_ratio > 1.02 and slope > 0) or (ma_ratio < 0.98 and slope < 0):
            return 1.15
        if (ma_ratio > 1.02 and slope < 0) or (ma_ratio < 0.98 and slope > 0):
            return 0.85
        return 1.0

    @staticmethod
    def detect_reversal(price_series, atr, volume, win=3, atr_mult=1.05, vol_mult=1.10) -> int:
        if len(price_series) < win + 1 or atr is None:
            return 0
        pct = np.diff(price_series) / price_series[:-1]
        slope_now, slope_prev = pct[-1], pct[-win:].mean()
        amp = max(price_series[-win - 1 :]) - min(price_series[-win - 1 :])
        price_base = price_series[-2] or price_series[-1]
        amp_pct = amp / price_base if price_base else 0
        cond_amp = amp_pct > atr_mult * atr
        cond_vol = (volume is None) or (volume >= vol_mult)
        if np.sign(slope_now) != np.sign(slope_prev) and cond_amp and cond_vol:
            return int(np.sign(slope_now))
        return 0

# utils/robust_scaler.py

import numpy as np
import json
import pandas as pd

def compute_robust_z_params(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    计算 df 中若干列的剪裁/缩放参数：
      lower: 0.25% 分位
      upper: 99.75% 分位
      mean : 平均值
      std  : 标准差（加 1e-6 避免除 0）
    返回形式：
    {
      "rsi_1h": {"lower":  10.123, "upper": 90.456, "mean": 50.789, "std": 12.345},
      ...
    }
    """
    scaler_params = {}
    for col in cols:
        arr = df[col].values
        arr_nonan = arr[~pd.isna(arr)]
        if len(arr_nonan) == 0:
            continue

        lower = np.percentile(arr_nonan, 0.25)
        upper = np.percentile(arr_nonan, 99.75)
        mean = np.mean(arr_nonan)
        std = np.std(arr_nonan) + 1e-6

        scaler_params[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "mean": float(mean),
            "std": float(std),
        }
    return scaler_params

def save_scaler_params_to_json(params: dict, filepath: str):
    """将 compute_robust_z_params 的结果写入 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

def load_scaler_params_from_json(filepath: str) -> dict:
    """从 JSON 文件加载剪裁/缩放参数。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_robust_z_with_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    用已知的剪裁/缩放参数，对 df 中对应列做 Robust-z：
      1. clip(lower, upper)
      2. (x - mean) / std
    """
    df_out = df.copy()
    for col, p in params.items():
        if col not in df_out.columns:
            continue
        lower = p["lower"]
        upper = p["upper"]
        mean = p["mean"]
        std = p["std"]
        df_out[col] = df_out[col].clip(lower, upper)
        df_out[col] = (df_out[col] - mean) / std
    return df_out

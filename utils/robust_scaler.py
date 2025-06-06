# utils/robust_scaler.py

import json
import numpy as np
import pandas as pd

def compute_robust_z_params(df: pd.DataFrame, cols: list) -> dict:
    """
    计算训练集所有数值特征列的 1% / 99% 分位、均值和标准差，并返回一个 dict：
    {
      "col1": {"lower": xx, "upper": xx, "mean": xx, "std": xx},
      "col2": {…}, …
    }
    """
    params = {}
    for col in cols:
        arr = df[col].dropna().values
        # ===== 从原来的 0.25%/99.75% 改为 1%/99% =====
        lower, upper = np.nanpercentile(arr, [1, 99])
        clipped = np.clip(arr, lower, upper)
        mu = float(np.nanmean(clipped))
        sigma = float(np.nanstd(clipped)) + 1e-6
        params[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "mean": mu,
            "std": sigma
        }
    return params

def save_scaler_params_to_json(params: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)

def load_scaler_params_from_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def apply_robust_z_with_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    对传入的 DataFrame（每列应与 params 中键对应）按给定的 lower/upper/mean/std 做剪裁 + 归一化，
    返回相同索引、已缩放好的 DataFrame。
    """
    df_scaled = df.copy()
    for col, p in params.items():
        if col not in df_scaled.columns:
            continue
        arr = df_scaled[col].values.astype(float)
        # 用训练时保存的 lower/upper 进行剪裁
        arr_clipped = np.clip(arr, p["lower"], p["upper"])
        # 再用训练时的 mean/std 归一化
        df_scaled[col] = (arr_clipped - p["mean"]) / p["std"]
    return df_scaled

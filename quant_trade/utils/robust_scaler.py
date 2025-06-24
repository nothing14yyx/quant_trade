# utils/robust_scaler.py

import json
import numpy as np
import pandas as pd


def compute_robust_z_params(df: pd.DataFrame, cols: list) -> dict:
    """
    计算训练集所有数值特征列的 0.5% / 99.5% 分位、均值和标准差，并返回一个 dict：
    {
      "col1": {"lower": xx, "upper": xx, "mean": xx, "std": xx},
      "col2": {…}, …
    }
    """
    params = {}
    for col in cols:
        arr = df[col].dropna().values
        if arr.size == 0:
            continue
        # ===== 从原来的 1%/99% 改为 0.5%/99.5%，更宽容地保留极值 =====
        lower, upper = np.nanpercentile(arr, [0.5, 99.5])
        clipped = np.clip(arr, lower, upper)
        mu = float(np.nanmean(clipped))
        sigma = float(np.nanstd(clipped)) + 1e-6
        params[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "mean": mu,
            "std": sigma,
        }
    return params


def save_scaler_params_to_json(params: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_scaler_params_from_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def apply_robust_z_with_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """根据给定参数进行剪裁与归一化，支持按 symbol 分组的参数结构。"""

    df_scaled = df.copy()

    # 若 params 的键不是列名，则视为 {symbol: {col: params}} 结构
    if not set(params.keys()) & set(df_scaled.columns):
        if "symbol" not in df_scaled.columns:
            raise ValueError("DataFrame 缺少 symbol 列，无法匹配分组参数")
        sym = df_scaled["symbol"].iloc[0]
        params = params.get(sym, {})

    for col, p in params.items():
        if col not in df_scaled.columns:
            continue
        arr = df_scaled[col].values.astype(float)
        arr_clipped = np.clip(arr, p["lower"], p["upper"])
        df_scaled[col] = (arr_clipped - p["mean"]) / p["std"]

    return df_scaled

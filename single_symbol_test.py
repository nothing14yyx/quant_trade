import os
import pytest
pytest.skip("requires database access", allow_module_level=True)

import pandas as pd
import yaml
from pathlib import Path
from sqlalchemy import create_engine
import numpy as np

from feature_engineering import (
    calc_features_raw,
    apply_robust_z_with_params,
    load_scaler_params_from_json,
    calc_cross_features,
)
from robust_signal_generator import RobustSignalGenerator

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"

# =================== 0. 切换开关 ===================
# 如果 USE_NORMALIZED = True，则对 calc_features_raw 的输出做 Robust‐Z 缩放后再传给模型
# 如果 USE_NORMALIZED = False，则直接把 calc_features_raw 的 raw 特征传给模型（不做缩放）
USE_NORMALIZED = True

# =================== 1. 读取配置 ===================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

mysql_cfg = config['mysql']
db_url = (
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg['port']}/{mysql_cfg['database']}?charset={mysql_cfg['charset']}"
)
engine = create_engine(db_url)

# =================== 2. 加载 Robust‐Z 缩放参数（仅在 USE_NORMALIZED=True 时需要） ===================
SCALER_PARAMS = None
if USE_NORMALIZED:
    scaler_path = config["feature_engineering"]["scaler_path"]
    SCALER_PARAMS = load_scaler_params_from_json(scaler_path)

# =================== 3. 读取原始 K 线（取最新 N 条） ===================
HISTORY_LEN = 200  # 加大窗口起点，确保所有技术指标都有足够历史
symbol = "UNIUSDT"

intervals = ["1h", "4h", "1d"]
dfs_raw = {}

for itv in intervals:
    dfs_raw[itv] = (
        pd.read_sql(
            f"SELECT * FROM klines WHERE symbol='{symbol}' AND `interval`='{itv}' ORDER BY open_time",
            engine,
            parse_dates=["open_time", "close_time"],
        )
        .set_index("open_time")
        .tail(HISTORY_LEN)
    )

# =================== 4. 计算原始 raw 特征 DataFrame ===================
# calc_features_raw 会返回完整的指标，包括 'atr_pct_1h'、'rsi_slope_1h' 等 raw 值
_f1h = calc_features_raw(dfs_raw["1h"], "1h")
_f4h = calc_features_raw(dfs_raw["4h"], "4h")
_fd1 = calc_features_raw(dfs_raw["1d"], "d1")

# 计算跨周期特征并与 1h 原始数据合并，添加时间字段
cross_feats = calc_cross_features(_f1h, _f4h, _fd1)
merged = (
    dfs_raw["1h"].reset_index()
    .merge(cross_feats, on="open_time", how="left", suffixes=("", "_feat"))
)
merged["symbol"] = symbol
merged["hour_of_day"] = merged["open_time"].dt.hour.astype(float)
merged["day_of_week"] = merged["open_time"].dt.dayofweek.astype(float)

raw_feats = merged

# =================== 5. 初始化信号生成器 ===================
model_paths = config["models"]
feature_cols_1h = config["feature_cols"]["1h"]
feature_cols_4h = config["feature_cols"]["4h"]
feature_cols_d1 = config["feature_cols"]["1d"]

signal_generator = RobustSignalGenerator(
    model_paths,
    feature_cols_1h=feature_cols_1h,
    feature_cols_4h=feature_cols_4h,
    feature_cols_d1=feature_cols_d1,
)

# =================== 6. 预处理并循环生成信号 ===================
scaled_feats = (
    apply_robust_z_with_params(raw_feats.copy(), SCALER_PARAMS)
    if USE_NORMALIZED
    else raw_feats.copy()
)

N = 60
for idx in range(-N, 0):
    row_scaled = scaled_feats.iloc[idx]
    row_raw = raw_feats.iloc[idx]

    feat_1h = {c: row_scaled[c] for c in feature_cols_1h if c in row_scaled}
    feat_4h = {c: row_scaled[c] for c in feature_cols_4h if c in row_scaled}
    feat_d1 = {c: row_scaled[c] for c in feature_cols_d1 if c in row_scaled}

    feat_1h["close"] = row_raw["close"]

    raw_dict = row_raw.to_dict()

    result = signal_generator.generate_signal(
        feat_1h,
        feat_4h,
        feat_d1,
        raw_features_1h=raw_dict,
        raw_features_4h=raw_dict,
        raw_features_d1=raw_dict,
    )

    print("=" * 60)
    print(f"时间 (1h): {row_raw['open_time']}")
    print("Feat 1h (最后一行)：", feat_1h)
    print("Feat 4h (最后一行)：", feat_4h)
    print("Feat d1 (最后一行)：", feat_d1)
    print("Signal Result:", result)

print("===== 单币种测试结束 =====")

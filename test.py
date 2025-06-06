import pandas as pd
import yaml
from sqlalchemy import create_engine
from utils.helper import calc_features_full  # 确保是新版 helper（全 float64）
from utils.robust_scaler import load_scaler_params_from_json, apply_robust_z_with_params

# === 1. 读取配置 ===
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
scaler_path = cfg["feature_engineering"]["scaler_path"]

# === 2. 数据库连接 ===
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}@"
    f"{mysql_cfg['host']}:{mysql_cfg.get('port', 3306)}/{mysql_cfg['database']}?"
    f"charset={mysql_cfg.get('charset', 'utf8mb4')}"
)

# === 3. 读取 K线数据（以 BTCUSDT 的 1h 为例） ===
symbol = "BTCUSDT"
interval = "1h"
sql = f"""
    SELECT * FROM klines
    WHERE symbol = '{symbol}' AND `interval` = '{interval}'
    ORDER BY open_time DESC
    LIMIT 100
"""
df = pd.read_sql(sql, engine, parse_dates=["open_time", "close_time"])
df = df.sort_values("open_time")  # 升序排列

# === 4. 特征工程（技术指标计算） ===
feats = calc_features_full(df, period=interval)

# === 5. Robust-z 标准化 ===
scaler_params = load_scaler_params_from_json(scaler_path)
feats_scaled = apply_robust_z_with_params(feats, scaler_params)

# === 6. 输出最后一行标准化后的特征（可用于推理） ===
print(feats_scaled.tail(1).T)

print(pd.__version__)

# 缺失值统计
print("每列缺失（为0/NaN）个数：")
print((feats == 0).sum())
print(feats.isna().sum())

# 极值分布
print("特征极值统计：")
print(feats.describe(percentiles=[.01, .05, .5, .95, .99]).T)

# _isnan列统计
nan_cols = [c for c in feats.columns if c.endswith('_isnan')]
print("各特征缺失情况：")
print(feats[nan_cols].sum())
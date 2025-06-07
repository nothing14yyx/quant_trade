import pandas as pd
import yaml
from pathlib import Path
from sqlalchemy import create_engine
import numpy as np
import pytest

# Skip this integration test when running in automated environments
# pytest.skip("requires database access", allow_module_level=True)

from feature_engineering import calc_features_raw, apply_robust_z_with_params, load_scaler_params_from_json
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
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}"
    f"@{mysql_cfg['host']}:{mysql_cfg['port']}/{mysql_cfg['database']}?charset={mysql_cfg['charset']}"
)
engine = create_engine(db_url)

# =================== 2. 加载 Robust‐Z 缩放参数（仅在 USE_NORMALIZED=True 时需要） ===================
SCALER_PARAMS = None
if USE_NORMALIZED:
    scaler_path = config["feature_engineering"]["scaler_path"]
    SCALER_PARAMS = load_scaler_params_from_json(scaler_path)

# =================== 3. 读取原始 K 线（取最新 60 条） ===================
symbol = "TAOUSDT"
intervals = ["1h", "4h", "1d"]
dfs_raw = {}

for itv in intervals:
    dfs_raw[itv] = pd.read_sql(
        f"SELECT * FROM klines WHERE symbol='{symbol}' AND `interval`='{itv}' ORDER BY open_time",
        engine,
        parse_dates=["open_time", "close_time"]
    ).tail(60).reset_index(drop=True)

# =================== 4. 计算原始 raw 特征 DataFrame ===================
# calc_features_raw 会返回完整的指标，包括 'atr_pct_1h'、'rsi_slope_1h' 等 raw 值
raw_feats = {
    "1h": calc_features_raw(dfs_raw["1h"], "1h").reset_index(drop=True),
    "4h": calc_features_raw(dfs_raw["4h"], "4h").reset_index(drop=True),
    "1d": calc_features_raw(dfs_raw["1d"], "d1").reset_index(drop=True),
}

# =================== 5. 初始化信号生成器 ===================
model_paths = config["models"]
feature_cols_1h = config["feature_cols"]["1h"]
feature_cols_4h = config["feature_cols"]["4h"]
feature_cols_d1 = config["feature_cols"]["d1"]

signal_generator = RobustSignalGenerator(
    model_paths,
    feature_cols_1h=feature_cols_1h,
    feature_cols_4h=feature_cols_4h,
    feature_cols_d1=feature_cols_d1,
)

# =================== 6. 循环取最后 N 根 K 线，生成信号 ===================
N = 60
for idx in range(-N, 0):
    # --- 6.1 如果需要归一化，则把 raw_feats 里的那一行先缩放 ---
    if USE_NORMALIZED:
        # 从 raw_feats 中取出“第 idx 行”（注意要保持 DataFrame 结构，用 [[idx]]）
        last_raw_1h = raw_feats["1h"].iloc[[idx]]
        last_raw_4h = raw_feats["4h"].iloc[[idx]]
        last_raw_d1 = raw_feats["1d"].iloc[[idx]]

        # 用 SCALER_PARAMS 做 Robust‐Z 缩放
        scaled_1h_df = apply_robust_z_with_params(last_raw_1h, SCALER_PARAMS)
        scaled_4h_df = apply_robust_z_with_params(last_raw_4h, SCALER_PARAMS)
        scaled_d1_df = apply_robust_z_with_params(last_raw_d1, SCALER_PARAMS)

        # 转为字典，作为模型输入
        feat_1h = scaled_1h_df.iloc[0].to_dict()
        feat_4h = scaled_4h_df.iloc[0].to_dict()
        feat_d1 = scaled_d1_df.iloc[0].to_dict()

        # 为了止盈止损，需要保证 feat_1h['close'] 和 feat_4h['close']
        # 均使用未缩放的原始收盘价
        feat_1h['close'] = dfs_raw["1h"]['close'].iloc[idx]
        feat_4h['close'] = dfs_raw["4h"]['close'].iloc[idx]

    else:
        # --- 6.2 如果不归一化，直接把 raw_feats 里的那一行转 dict ---
        feat_1h = raw_feats["1h"].iloc[idx].to_dict()
        feat_4h = raw_feats["4h"].iloc[idx].to_dict()
        feat_d1 = raw_feats["1d"].iloc[idx].to_dict()

        # raw_feats 里本身就包含 'close'（原始收盘价），不需要再覆盖

    # --- 6.3 创建原始特征字典，用于计算止盈止损等 ---
    raw_1h = raw_feats["1h"].iloc[idx].to_dict()
    raw_4h = raw_feats["4h"].iloc[idx].to_dict()
    raw_d1 = raw_feats["1d"].iloc[idx].to_dict()

    # --- 6.4 生成信号（需传入 raw 特征以获得正确的止盈止损） ---
    result = signal_generator.generate_signal(
        feat_1h, feat_4h, feat_d1,
        raw_features_1h=raw_1h,
        raw_features_4h=raw_4h,
        raw_features_d1=raw_d1
    )

    # --- 6.5 打印结果，方便比对 ---
    print("="*60)
    print(f"时间 (1h): {dfs_raw['1h']['open_time'].iloc[idx]}")
    print("Feat 1h (最后一行)：", feat_1h)
    print("Feat 4h (最后一行)：", feat_4h)
    print("Feat d1 (最后一行)：", feat_d1)
    print("Signal Result:", result)

print("===== 单币种测试结束 =====")

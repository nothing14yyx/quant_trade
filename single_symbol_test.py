import pandas as pd
import yaml
from sqlalchemy import create_engine
from robust_signal_generator import RobustSignalGenerator  # 你的信号生成器类
from feature_engineering import calc_features_full        # 你的特征工程函数

# 读取config
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 数据库连接
mysql_cfg = config['mysql']
db_url = f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}@{mysql_cfg['host']}:{mysql_cfg['port']}/{mysql_cfg['database']}?charset={mysql_cfg['charset']}"
engine = create_engine(db_url)

# 配置周期
symbol = "BTCUSDT"    # 你要测试的币种
intervals = ["1h", "4h", "1d"]

dfs = {}
for itv in intervals:
    dfs[itv] = pd.read_sql(
        f"SELECT * FROM klines WHERE symbol='{symbol}' AND `interval`='{itv}' ORDER BY open_time",
        engine,
        parse_dates=["open_time", "close_time"]
    )

# 特征工程（只算最新60根）
for itv in intervals:
    dfs[itv] = dfs[itv].tail(60)
feats_1h_df = calc_features_full(dfs["1h"], "1h")
feats_4h_df = calc_features_full(dfs["4h"], "4h")
feats_d1_df = calc_features_full(dfs["1d"], "d1")

# 加载模型与特征列
model_paths = config['models']
feature_cols_1h = config['feature_cols']['1h']
feature_cols_4h = config['feature_cols']['4h']
feature_cols_d1 = config['feature_cols']['d1']
signal_generator = RobustSignalGenerator(
    model_paths, feature_cols_1h, feature_cols_4h, feature_cols_d1
)

# 逐步测试最后N根，N可调（如20）
N = 20
for i in range(-N, 0):
    feat_1h = feats_1h_df.iloc[i].to_dict()
    feat_4h = feats_4h_df.iloc[i].to_dict()
    feat_d1 = feats_d1_df.iloc[i].to_dict()
    feat_4h['close'] = dfs["4h"]["close"].iloc[i]   # 保证有close用于止盈止损
    # 生成信号
    result = signal_generator.generate_signal(feat_1h, feat_4h, feat_d1)
    print("="*60)
    print(f"时间: {dfs['1h']['open_time'].iloc[i]}")
    print("特征(1h):", feat_1h)
    print("特征(4h):", feat_4h)
    print("特征(d1):", feat_d1)
    print("信号结果:", result)

    # 如需写csv便于对比
    # pd.DataFrame([result]).to_csv("debug_signal_results.csv", mode='a', index=False, header=(i==-N))

print("单币种特征与信号检查结束。")

import pandas as pd
from quant_trade.utils.db import load_config, connect_mysql
from sqlalchemy import text

cfg = load_config()          # 读取 config.yaml
engine = connect_mysql(cfg)  # 创建数据库连接

n = 10000  # 需要导出的行数
query = text("SELECT * FROM features ORDER BY open_time DESC LIMIT :n")
df = pd.read_sql(query, engine, params={"n": n},
                 parse_dates=["open_time", "close_time"])

df.to_csv("features_recent.csv", index=False)  # 保存为 CSV

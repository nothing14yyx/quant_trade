import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://user:pwd@host/db")  # 用实际连接串
with engine.begin() as conn:
    cols = conn.execute(text("SHOW COLUMNS FROM features")).fetchall()
print("features 表当前列数:", len(cols))
print("有没有 ichimoku_cloud_thickness_d1 ?",
      any(c[0] == "ichimoku_cloud_thickness_d1" for c in cols))

from pathlib import Path

import pandas as pd
from quant_trade.utils.db import load_config, connect_mysql
from sqlalchemy import text

cfg = load_config()          # 读取 config.yaml
engine = connect_mysql(cfg)  # 创建数据库连接

n = 10000  # 需要导出的行数
query = text("SELECT * FROM features ORDER BY open_time DESC LIMIT :n")
df = pd.read_sql(query, engine, params={"n": n},
                 parse_dates=["open_time", "close_time"])

OUT_DIR = Path(__file__).resolve().parent / "exp_feature"
OUT_DIR.mkdir(exist_ok=True)
temp_file = OUT_DIR / "features_recent.csv"
df.to_csv(temp_file, index=False)  # 保存为 CSV

def split_csv(file_path: Path, out_dir: Path, max_mb: int = 20) -> None:
    """将 CSV 文件按大小拆分存入目标文件夹"""
    max_bytes = max_mb * 1024 * 1024
    with file_path.open("r", encoding="utf-8") as src:
        header = src.readline()
        part = 1
        dst = (out_dir / f"{file_path.stem}_part{part}.csv").open(
            "w", encoding="utf-8"
        )
        dst.write(header)
        size = len(header.encode("utf-8"))
        for line in src:
            line_size = len(line.encode("utf-8"))
            if size + line_size > max_bytes:
                dst.close()
                part += 1
                dst = (out_dir / f"{file_path.stem}_part{part}.csv").open(
                    "w", encoding="utf-8"
                )
                dst.write(header)
                size = len(header.encode("utf-8"))
            dst.write(line)
            size += line_size
        dst.close()
    file_path.unlink(missing_ok=True)


split_csv(temp_file, OUT_DIR)

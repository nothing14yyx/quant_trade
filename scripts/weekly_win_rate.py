import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
import yaml


def load_data(csv_path: str = None, db_uri: str = None, table: str = "live_full_data"):
    """根据 csv 或数据库 URI 读取交易数据"""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif db_uri:
        engine = create_engine(db_uri)
        df = pd.read_sql_table(table, engine)
    else:
        raise ValueError("必须提供 csv 路径或数据库 URI")
    return df


def filter_recent_week(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留最近七天的数据"""
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    time_col = "entry_time" if "entry_time" in df.columns else "time"
    df[time_col] = pd.to_datetime(df[time_col])
    return df[df[time_col] >= week_ago]


def calc_win_rate(df: pd.DataFrame) -> tuple[float, int]:
    """根据 ret>0 计算胜率和总数"""
    total = len(df)
    wins = (df.get("ret", 0) > 0).sum()
    win_rate = wins / total if total else 0
    return win_rate, total


def main():
    parser = argparse.ArgumentParser(description="统计最近七天的胜率")
    parser.add_argument("--csv", help="交易记录 csv 路径", default="backtest_fusion_trades_all.csv")
    parser.add_argument("--db", help="数据库 URI，如 mysql+pymysql://user:pwd@host/db")
    parser.add_argument("--config", help="配置文件路径，若提供则从中读取 database.uri")
    args = parser.parse_args()

    db_uri = args.db
    if args.config and not db_uri and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            db_uri = cfg.get("database", {}).get("uri")

    df = load_data(args.csv if os.path.exists(args.csv) else None, db_uri)
    df_recent = filter_recent_week(df)
    win_rate, total = calc_win_rate(df_recent)
    print(f"最近7天总交易数: {total}，胜率: {win_rate:.2%}")


if __name__ == "__main__":
    main()

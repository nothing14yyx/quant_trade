#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
pytest.skip("integration helper", allow_module_level=True)
"""
feature_checks.py

用于检查刚生成的 feature 表（CSV 或 MySQL）是否符合预期，包括：
1. 列名与数据类型检查
2. 缺失值和 Inf 值统计
3. 时间序列连续性检查（以 1h 周期为例）
4. target_up/target_down 标签分布
5. 部分指标基本统计量
6. 简单时序 CV 验证（可选）

现在直接从 utils/config.yaml 里读取 MySQL 配置，无需手动传入连接字符串。

使用方法：
    # 从 config.yaml 里读 MySQL 并检查 features 表
    python feature_checks.py --source mysql

    # 或直接检查本地 CSV
    python feature_checks.py --source csv --path /path/to/features.csv

    # 若只做前五步检查，跳过 LightGBM 验证，加上 --skip-cv
    python feature_checks.py --source mysql --skip-cv
"""

import argparse
import sys
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_mysql_conn_str(mysql_cfg: dict) -> str:
    """
    根据 config 中的 mysql 配置，拼接 SQLAlchemy 连接字符串
    mysql_cfg 形如 {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "123456",
        "database": "quant_trading",
        "charset": "utf8mb4"
    }
    """
    user = mysql_cfg.get("user", "")
    pw = mysql_cfg.get("password", "")
    host = mysql_cfg.get("host", "localhost")
    port = mysql_cfg.get("port", 3306)
    db = mysql_cfg.get("database", "")
    charset = mysql_cfg.get("charset", "utf8mb4")
    return f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset={charset}"


def load_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    return df


def load_data_from_mysql_via_config(config_path: str, table: str) -> pd.DataFrame:
    """从 utils/config.yaml 中读取 MySQL 配置，然后载入指定表"""
    cfg = load_config(config_path)
    mysql_cfg = cfg.get("mysql")
    if mysql_cfg is None:
        raise RuntimeError(f"{config_path} 中未找到 mysql 配置。")
    conn_str = make_mysql_conn_str(mysql_cfg)
    from sqlalchemy import create_engine
    engine = create_engine(conn_str)
    df = pd.read_sql(f"SELECT * FROM `{table}`", engine, parse_dates=["open_time", "close_time"])
    return df


def check_dtypes(df: pd.DataFrame):
    print("\n=== 1. 数据类型检查 ===")
    print(df.dtypes)


def check_missing_inf(df: pd.DataFrame):
    print("\n=== 2. 缺失值＆Inf 值统计 ===")
    null_counts = df.isna().sum().sort_values(ascending=False)
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print("存在缺失值的列：")
        print(null_counts)
    else:
        print("所有列均无缺失值。")

    inf_counts = ((df == float("inf")).sum() + (df == float("-inf")).sum()).sort_values(ascending=False)
    inf_counts = inf_counts[inf_counts > 0]
    if not inf_counts.empty:
        print("\n存在 Inf / -Inf 的列：")
        print(inf_counts)
    else:
        print("所有列均无 Inf 或 -Inf。")


def check_time_continuity(df: pd.DataFrame):
    print("\n=== 3. 时间序列连续性检查 (1h 周期，按 symbol 分组) ===")
    if "open_time" not in df.columns or "symbol" not in df.columns:
        print("错误：没有找到 'open_time' 或 'symbol' 列，无法检查时间连续性。")
        return

    non_consecutive = {}
    # 按 symbol 分组逐个检查
    for sym, grp in df.groupby("symbol"):
        grp_sorted = grp.sort_values("open_time").reset_index(drop=True)
        diffs = grp_sorted["open_time"].diff().dropna()
        # 找出所有不等于 1 小时的差值
        bad = diffs[diffs != pd.Timedelta(hours=1)]
        if not bad.empty:
            non_consecutive[sym] = len(bad)

    if not non_consecutive:
        print("每个币种的 open_time 都严格按 1 小时递增。")
    else:
        print("以下币种存在非 1 小时的时间间隔：")
        for sym, cnt in non_consecutive.items():
            print(f"  {sym}：{cnt} 处不连续")


def check_label_distribution(df: pd.DataFrame):
    print("\n=== 4. 标签分布检查 ===")
    for col in ["target_up", "target_down"]:
        if col not in df.columns:
            print(f"警告：没有找到 '{col}' 列。")
            continue
        print(f"\n{col} 取值统计：")
        print(df[col].value_counts(dropna=False))
        nan_count = df[col].isna().sum()
        print(f"  NaN 数量：{nan_count}")


def check_basic_stats(df: pd.DataFrame):
    print("\n=== 5. 部分指标基本统计量 ===")
    # 收集所有以 _1h、_4h 或 _d1 结尾且不包含 _isnan 的列
    cols_to_describe = []
    for col in df.columns:
        if (col.endswith("_1h") or col.endswith("_4h") or col.endswith("_d1")) and "_isnan" not in col:
            cols_to_describe.append(col)

    if not cols_to_describe:
        print("没有发现以 _1h、_4h 或 _d1 结尾的指标列。")
        return

    # 临时调整 Pandas 打印设置，显示所有行列
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)

    desc = df[cols_to_describe].describe().T
    print(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    # 恢复 Pandas 默认打印设置，防止影响后续输出
    pd.reset_option("display.max_columns")
    pd.reset_option("display.max_rows")
    pd.reset_option("display.width")


def check_new_features(df: pd.DataFrame):
    print("\n=== 6. 新增特征检查 ===")
    required_cols = [
        "close_spread_1h_4h", "close_spread_1h_d1",
        "ma_ratio_1h_4h", "ma_ratio_1h_d1",
        "hour_of_day", "day_of_week",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("缺失列：", missing)
    else:
        stats = df[required_cols].describe().loc[["count", "mean"]]
        print(stats)


def simple_time_series_cv(df: pd.DataFrame, prefix: str = "1h"):
    print("\n=== 6. 简单时序 CV 验证（LightGBM） ===")
    feature_cols = [c for c in df.columns if c.endswith(f"_{prefix}") or c.endswith(f"_{prefix}_isnan")]
    if not feature_cols:
        print(f"没有找到以 _{prefix} 或 _{prefix}_isnan 结尾的特征列，跳过 CV 验证。")
        return

    if "target_up" not in df.columns:
        print("没有找到 'target_up'，无法做二分类 CV。")
        return

    df2 = df.dropna(subset=["target_up"])
    X = df2[feature_cols]
    y = df2["target_up"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = LGBMClassifier(n_estimators=100, random_state=42)
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        p = m.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, p)
        aucs.append(auc)

    print("5 折时序 CV 得分：", np.round(aucs, 4))
    print("平均 AUC：", np.round(np.mean(aucs), 4))


def main():
    parser = argparse.ArgumentParser(description="检查 feature 表是否正常")
    parser.add_argument(
        "--source", choices=["csv", "mysql"], required=True,
        help="数据源：csv 或 mysql"
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="若 source=csv，则指定 CSV 文件路径"
    )
    parser.add_argument(
        "--table", type=str, default="features",
        help="若 source=mysql，则指定表名，默认为 features"
    )
    parser.add_argument(
        "--config", type=str, default="utils/config.yaml",
        help="指定 config.yaml 路径，默认为 utils/config.yaml"
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="若指定此标志，则跳过第 6 步的简单时序 CV 验证"
    )
    args = parser.parse_args()

    # 读取数据
    if args.source == "csv":
        if not args.path:
            print("错误：请用 --path 指定 CSV 文件路径。")
            sys.exit(1)
        print(f"读取 CSV 数据：{args.path}")
        df = load_data_from_csv(args.path)
    else:
        # 直接从 config.yaml 中读取 MySQL 配置并载入表
        print("从 MySQL 读取数据，配置文件：", args.config, "表名：", args.table)
        df = load_data_from_mysql_via_config(args.config, args.table)

    print("\n数据读取完毕，记录数：", df.shape[0], "列数：", df.shape[1])

    # 1. 数据类型检查
    check_dtypes(df)

    # 2. 缺失值 & Inf 值统计
    check_missing_inf(df)

    # 3. 时间序列连续性检查
    check_time_continuity(df)

    # 4. target_up / target_down 标签分布
    check_label_distribution(df)

    # 5. 部分指标基本统计量
    check_basic_stats(df)

    # 6. 新增特征检查
    check_new_features(df)

    # 7. 简单时序 CV 验证
    if not args.skip_cv:
        simple_time_series_cv(df, prefix="1h")


if __name__ == "__main__":
    main()

import os

import logging

from pathlib import Path
import pandas as pd
import yaml
from sqlalchemy import create_engine

from robust_signal_generator import RobustSignalGenerator
from utils.helper import calc_features_raw

from feature_engineering import calc_cross_features

from utils.robust_scaler import (
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)



def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_mysql(cfg):
    mysql = cfg["mysql"]
    url = (
        f"mysql+pymysql://{mysql['user']}:{os.getenv('MYSQL_PASSWORD', mysql['password'])}"
        f"@{mysql['host']}:{mysql.get('port', 3306)}/{mysql['database']}?charset=utf8mb4"
    )
    return create_engine(url)


def load_latest_klines(engine, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    query = (
        "SELECT open_time, open, high, low, close, volume, fg_index, funding_rate "
        f"FROM klines WHERE symbol='{symbol}' AND `interval`='{interval}' "
        f"ORDER BY open_time DESC LIMIT {limit}"
    )
    df = pd.read_sql(query, engine, parse_dates=["open_time"])
    return df.sort_values("open_time")


def prepare_features(df: pd.DataFrame, period: str, params: dict, symbol: str) -> tuple[dict, dict]:

    """计算单周期特征并返回(缩放后的dict, 原始dict)"""

    feats = calc_features_raw(df.set_index("open_time"), period)
    raw = feats.iloc[-1]


    last = feats.tail(1).copy()
    last["symbol"] = symbol
    last = apply_robust_z_with_params(last, params)
    scaled = last.drop(columns=["symbol"]).iloc[0]
    return scaled.to_dict(), raw.to_dict()



def prepare_all_features(engine, symbol: str, params: dict) -> tuple[dict, dict, dict, dict, dict, dict]:
    """加载多周期K线并构造包含跨周期特征的字典"""

    df1h = load_latest_klines(engine, symbol, "1h")
    df4h = load_latest_klines(engine, symbol, "4h")
    dfd1 = load_latest_klines(engine, symbol, "1d")

    f1h_df = calc_features_raw(df1h.set_index("open_time"), "1h")
    f4h_df = calc_features_raw(df4h.set_index("open_time"), "4h")
    fd1_df = calc_features_raw(dfd1.set_index("open_time"), "d1")

    merged = calc_cross_features(f1h_df, f4h_df, fd1_df)
    merged["hour_of_day"] = merged["open_time"].dt.hour.astype(float)
    merged["day_of_week"] = merged["open_time"].dt.dayofweek.astype(float)

    cross_last = merged.tail(1).copy()
    cross_last["symbol"] = symbol
    cross_scaled = apply_robust_z_with_params(cross_last, params).drop(columns=["symbol"]).iloc[0]

    cross_cols = [c for c in cross_last.columns if any(x in c for x in ["_1h_4h", "_1h_d1", "_4h_d1", "hour_of_day", "day_of_week"])]

    scaled1h, raw1h = prepare_features(df1h, "1h", params, symbol)
    scaled4h, raw4h = prepare_features(df4h, "4h", params, symbol)
    scaledd1, rawd1 = prepare_features(dfd1, "d1", params, symbol)

    for col in cross_cols:
        scaled1h[col] = cross_scaled[col]
        raw1h[col] = cross_last[col].iloc[0]

    return scaled1h, scaled4h, scaledd1, raw1h, raw4h, rawd1



def main(symbol: str = "BTCUSDT"):
    cfg = load_config()
    engine = connect_mysql(cfg)

    params = load_scaler_params_from_json(cfg["feature_engineering"]["scaler_path"])


    feats1h, feats4h, featsd1, raw1h, raw4h, rawd1 = prepare_all_features(engine, symbol, params)


    sg = RobustSignalGenerator(
        model_paths=cfg["models"],
        feature_cols_1h=cfg.get("feature_cols", {}).get("1h", []),
        feature_cols_4h=cfg.get("feature_cols", {}).get("4h", []),
        feature_cols_d1=cfg.get("feature_cols", {}).get("1d", []),
    )

    signal = sg.generate_signal(
        feats1h,
        feats4h,
        featsd1,
        raw_features_1h=raw1h,
        raw_features_4h=raw4h,
        raw_features_d1=rawd1,
        symbol=symbol,
    )


    logging.info("最新交易信号:\n%s", signal)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从数据库获取数据生成交易信号")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易对，如 BTCUSDT")
    args = parser.parse_args()
    main(args.symbol)


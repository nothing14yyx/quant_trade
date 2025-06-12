import argparse
import yaml
import pandas as pd
from data_loader import DataLoader
from robust_signal_generator import RobustSignalGenerator


def load_config(path: str = "utils/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_cols(path: str = "utils/selected_features.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_latest_features(engine, symbol: str, period: str, cols: list[str]):
    col_str = ",".join(f"`{c}`" for c in cols)
    query = (
        f"SELECT {col_str} FROM features "
        f"WHERE symbol=:s AND `interval`=:iv ORDER BY open_time DESC LIMIT 1"
    )
    df = pd.read_sql(query, engine, params={"s": symbol, "iv": period})
    return df.iloc[0].to_dict() if not df.empty else {}


def main(symbol: str):
    cfg = load_config()
    loader = DataLoader()
    feature_cols = load_feature_cols()

    model_paths = cfg.get("models", {})
    rsg = RobustSignalGenerator(
        model_paths,
        feature_cols_1h=feature_cols.get("1h", []),
        feature_cols_4h=feature_cols.get("4h", []),
        feature_cols_d1=feature_cols.get("1d", []),
    )

    feats_1h = get_latest_features(loader.engine, symbol, "1h", feature_cols.get("1h", []))
    feats_4h = get_latest_features(loader.engine, symbol, "4h", feature_cols.get("4h", []))
    feats_d1 = get_latest_features(loader.engine, symbol, "1d", feature_cols.get("1d", []))

    oi = loader.get_latest_open_interest(symbol)
    obi = loader.get_latest_order_book_imbalance(symbol)
    global_metrics = loader.get_latest_cg_global_metrics()

    result = rsg.generate_signal(
        feats_1h,
        feats_4h,
        feats_d1,
        open_interest=oi,
        order_book_imbalance=obi,
        global_metrics=global_metrics,
        symbol=symbol,
    )

    print("=== Signal ===")
    print(result)

    print("\n=== Indicators ===")
    indicators = {
        "open_interest": oi,
        "order_book_imbalance": obi,
        "global_metrics": global_metrics,
    }
    print(indicators)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单币交易信号与指标查询")
    parser.add_argument("symbol", help="币安交易对，如 BTCUSDT")
    args = parser.parse_args()
    main(args.symbol.upper())

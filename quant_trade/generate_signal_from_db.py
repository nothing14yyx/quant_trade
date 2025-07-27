"""Generate trading signals from recent data in the database."""

from __future__ import annotations

import argparse
import logging
import pandas as pd

from quant_trade.utils.db import load_config, connect_mysql
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.utils.robust_scaler import load_scaler_params_from_json
from quant_trade.feature_loader import (
    load_latest_klines,
    prepare_all_features,
    load_latest_open_interest,
    load_order_book_imbalance,
    load_global_metrics,
    load_symbol_categories,
)


logger = logging.getLogger(__name__)


def main(symbol: str = "ETHUSDT") -> None:
    cfg = load_config()
    engine = connect_mysql(cfg)

    recent = load_latest_klines(engine, symbol, "1h", limit=20)
    params = load_scaler_params_from_json(cfg["feature_engineering"]["scaler_path"])

    global_metrics = load_global_metrics(engine, symbol)
    oi = load_latest_open_interest(engine, symbol)
    order_imb = load_order_book_imbalance(engine, symbol)
    categories = load_symbol_categories(engine)

    rsg_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
    sg = RobustSignalGenerator(rsg_cfg)
    sg.set_symbol_categories(categories)

    results = []
    latest_signal = None
    for idx in range(len(recent)):
        feats1h, feats4h, featsd1, raw1h, raw4h, rawd1 = prepare_all_features(
            engine, symbol, params, idx
        )
        if order_imb is not None and idx == 0:
            raw1h["bid_ask_imbalance"] = order_imb
        sig = sg.generate_signal(
            feats1h,
            feats4h,
            featsd1,
            raw_features_1h=raw1h,
            raw_features_4h=raw4h,
            raw_features_d1=rawd1,
            global_metrics=global_metrics,
            open_interest=oi,
            order_book_imbalance=order_imb if idx == 0 else None,
            symbol=symbol,
        )
        if sig is None:
            logger.debug("generate_signal returned None for idx %s", idx)
            continue
        if latest_signal is None:
            latest_signal = sig
        results.append(
            {
                "open_time": recent.iloc[-1 - idx]["open_time"],
                "close": recent.iloc[-1 - idx]["close"],
                "score": sig.get("score"),
            }
        )

    logger.info("最新交易信号:\n%s", latest_signal)
    logger.info("%s", pd.DataFrame(results).to_string(index=False))
    if hasattr(sg, "stop_weight_update_thread"):
        sg.stop_weight_update_thread()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="从数据库获取数据生成交易信号")
    parser.add_argument("--symbol", default="XRPUSDT", help="交易对，如 BTCUSDT")
    args = parser.parse_args()
    main(args.symbol)

"""Generate trading signals from recent data in the database."""

from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd

from quant_trade.utils.db import load_config, connect_mysql
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.risk_manager import RiskManager
from quant_trade.utils.robust_scaler import load_scaler_params_from_json
from quant_trade.feature_loader import (
    load_latest_klines,
    prepare_all_features,
    load_latest_open_interest,
    load_order_book_imbalance,
    load_global_metrics,
    load_symbol_categories,
)
from quant_trade.logging import get_logger
from quant_trade.signal.decision import DecisionConfig, decide_signal
from quant_trade.calibration import apply_temperature, TemperatureModel


logger = get_logger(__name__)


def main(symbol: str = "ETHUSDT") -> None:
    cfg = load_config()
    engine = connect_mysql(cfg)
    dec_cfg = DecisionConfig.from_dict(cfg.get("signal", {}))
    temp_model = TemperatureModel(cfg.get("calibration", {}).get("temperature", 1.0))

    recent = load_latest_klines(engine, symbol, "1h", limit=20)
    params = load_scaler_params_from_json(cfg["feature_engineering"]["scaler_path"])

    global_metrics = load_global_metrics(engine, symbol)
    oi = load_latest_open_interest(engine, symbol)
    order_imb = load_order_book_imbalance(engine, symbol)
    categories = load_symbol_categories(engine)

    rsg_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
    sg = RobustSignalGenerator(rsg_cfg)
    sg.risk_manager = RiskManager(**cfg.get("risk_manager", {}))
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
        details = sig.get("details") or {}

        probs = sig.get("probs")
        if probs is None:
            logits = sig.get("logits")
            if logits is not None:
                probs = apply_temperature(np.asarray(logits), temp_model)
            else:
                score_val = sig.get("score", 0.0)
                try:
                    score_val = float(score_val)
                except (TypeError, ValueError):
                    score_val = 0.0
                if not np.isfinite(score_val):
                    score_val = 0.0
                p_up = (score_val + 1.0) / 2.0
                probs = np.array([1 - p_up, 0.0, p_up])

        decision = decide_signal(
            probs,
            (details.get("rise_preds") or {}).get("1h"),
            (details.get("drawdown_preds") or {}).get("1h"),
            (details.get("vol_preds") or {}).get("1h"),
            bool(details.get("flip")),
            dec_cfg,
        )

        if latest_signal is None:
            latest_signal = decision
        results.append(
            {
                "open_time": recent.iloc[-1 - idx]["open_time"],
                "close": recent.iloc[-1 - idx]["close"],
                "score": sig.get("score"),
                "action": decision["action"],
                "size": decision["size"],
                "note": decision["note"],
            }
        )

    logger.info("最新交易信号:\n%s", latest_signal)
    logger.info("%s", pd.DataFrame(results).to_string(index=False))
    if hasattr(sg, "update_weights"):
        sg.update_weights()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="从数据库获取数据生成交易信号")
    parser.add_argument("--symbol", default="SOLUSDT", help="交易对，如 BTCUSDT")
    args = parser.parse_args()
    main(args.symbol)

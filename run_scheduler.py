# -*- coding: utf-8 -*-
"""Simple scheduler for periodic data sync and signal generation."""

import json
import logging
import time
from datetime import datetime, timedelta

from data_loader import DataLoader
from generate_signal_from_db import (
    load_config,
    connect_mysql,
    load_scaler_params_from_json,
    prepare_all_features,
    load_global_metrics,
    load_latest_open_interest,
    load_order_book_imbalance,
    load_symbol_categories,
)
from robust_signal_generator import RobustSignalGenerator
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Scheduler:
    def __init__(self) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.dl = DataLoader()
        self.engine = connect_mysql(cfg)
        self.scaler_params = load_scaler_params_from_json(
            cfg["feature_engineering"]["scaler_path"]
        )
        self.sg = RobustSignalGenerator(
            model_paths=cfg["models"],
            feature_cols_1h=cfg.get("feature_cols", {}).get("1h", []),
            feature_cols_4h=cfg.get("feature_cols", {}).get("4h", []),
            feature_cols_d1=cfg.get("feature_cols", {}).get("1d", []),
        )
        categories = load_symbol_categories(self.engine)
        self.sg.set_symbol_categories(categories)

    def update_klines(self, symbols, interval):
        for sym in symbols:
            try:
                self.dl.incremental_update_klines(sym, interval)
            except Exception as e:
                logging.exception("update %s %s failed: %s", sym, interval, e)

    def update_daily_data(self, symbols):
        try:
            self.dl.update_sentiment()
        except Exception as e:
            logging.exception("update sentiment failed: %s", e)
        try:
            self.dl.update_cg_global_metrics()
            self.dl.update_cg_market_data(symbols)
            self.dl.update_cg_coin_categories(symbols)
            self.dl.update_cg_category_stats()
        except Exception as e:
            logging.exception("update coingecko failed: %s", e)

    def generate_signals(self, symbols):
        logging.info("generating signals for %s symbols", len(symbols))
        global_metrics = load_global_metrics(self.engine)
        results = []
        now = datetime.utcnow().replace(second=0, microsecond=0)
        for sym in symbols:
            try:
                feats1h, feats4h, featsd1, raw1h, raw4h, rawd1 = prepare_all_features(
                    self.engine, sym, self.scaler_params
                )
                oi = load_latest_open_interest(self.engine, sym)
                order_imb = load_order_book_imbalance(self.engine, sym)
                sig = self.sg.generate_signal(
                    feats1h,
                    feats4h,
                    featsd1,
                    raw_features_1h=raw1h,
                    raw_features_4h=raw4h,
                    raw_features_d1=rawd1,
                    global_metrics=global_metrics,
                    open_interest=oi,
                    order_book_imbalance=order_imb,
                    symbol=sym,
                )
                data = {
                    "symbol": sym,
                    "time": now,
                    "price": feats1h.get("close"),
                    "signal": sig.get("signal"),
                    "score": sig.get("score"),
                    "pos": sig.get("position_size"),
                    "take_profit": sig.get("take_profit"),
                    "stop_loss": sig.get("stop_loss"),
                    "indicators": json.dumps(
                        {
                            "feat_1h": raw1h,
                            "feat_4h": raw4h,
                            "feat_d1": rawd1,
                            "details": sig.get("details"),
                        }
                    ),
                }
                results.append(data)
            except Exception as e:
                logging.exception("signal for %s failed: %s", sym, e)
        if not results:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO live_full_data "
                    "(symbol,time,price,signal,score,pos,take_profit,stop_loss,indicators) "
                    "VALUES (:symbol,:time,:price,:signal,:score,:pos,:take_profit,:stop_loss,:indicators)"
                ),
                results,
            )
        results.sort(key=lambda x: abs(x.get("score") or 0), reverse=True)
        top10 = results[:10]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO live_top10_signals "
                    "(symbol,time,price,signal,score,pos,take_profit,stop_loss,indicators) "
                    "VALUES (:symbol,:time,:price,:signal,:score,:pos,:take_profit,:stop_loss,:indicators)"
                ),
                top10,
            )

    def run(self):
        next_symbols_refresh = datetime.utcnow()
        symbols = []
        while True:
            now = datetime.utcnow()
            if now >= next_symbols_refresh:
                symbols = self.dl.get_top_symbols()
                next_symbols_refresh = now + timedelta(hours=1)
            minute = now.minute
            if minute % 15 == 0:
                self.update_klines(symbols, "5m")
            if minute % 30 == 0:
                self.update_klines(symbols, "15m")
            if minute == 0:
                self.update_klines(symbols, "1h")
                if now.hour % 4 == 0:
                    self.update_klines(symbols, "4h")
                if now.hour == 0:
                    self.update_klines(symbols, "1d")
                    self.update_daily_data(symbols)
                self.generate_signals(symbols)
            # sleep until next minute
            time.sleep(60 - datetime.utcnow().second)


if __name__ == "__main__":
    Scheduler().run()

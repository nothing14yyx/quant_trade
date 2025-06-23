# -*- coding: utf-8 -*-
"""Simple scheduler for periodic data sync and signal generation."""

import json
import math
import logging
import time
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
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
from utils.helper import collect_feature_cols
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _to_builtin(v):
    """Convert numpy scalar types to builtin Python types and handle NaN."""
    try:
        if np.isnan(v):
            return None
    except Exception:
        pass

    if isinstance(v, (np.bool_, np.integer)):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


_json_dumps_orig = json.dumps


def _dumps_with_nan(obj, *args, **kwargs):
    """Helper to encode NaN as null when default=_to_builtin."""
    if kwargs.get("default") is _to_builtin:
        kwargs.setdefault("allow_nan", False)
        try:
            return _json_dumps_orig(obj, *args, **kwargs)
        except ValueError:
            def _replace(o):
                if isinstance(o, float) and math.isnan(o):
                    return None
                elif isinstance(o, dict):
                    return {k: _replace(v) for k, v in o.items()}
                elif isinstance(o, (list, tuple)):
                    return [ _replace(v) for v in o ]
                return o
            obj = _replace(obj)
            return _json_dumps_orig(obj, *args, **kwargs)
    return _json_dumps_orig(obj, *args, **kwargs)


def safe_json_dumps(obj, *args, **kwargs):
    """Serialize JSON while handling NaN and numpy types."""
    kwargs.setdefault("default", _to_builtin)
    return _dumps_with_nan(obj, *args, **kwargs)


class Scheduler:
    def __init__(self) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.dl = DataLoader()
        self.fe = FeatureEngineer()
        self.engine = connect_mysql(cfg)
        self.scaler_params = load_scaler_params_from_json(
            cfg["feature_engineering"]["scaler_path"]
        )
        self.sg = RobustSignalGenerator(
            model_paths=cfg["models"],
            feature_cols_1h=collect_feature_cols(cfg, "1h"),
            feature_cols_4h=collect_feature_cols(cfg, "4h"),
            feature_cols_d1=collect_feature_cols(cfg, "d1"),
        )
        categories = load_symbol_categories(self.engine)
        self.sg.set_symbol_categories(categories)
        # 调度器与线程池，用于更精确和并发地执行任务
        import sched
        from concurrent.futures import ThreadPoolExecutor

        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.symbols = []
        self.next_symbols_refresh = datetime.now(UTC)
        self.ic_update_limit = cfg.get("ic_update_rows", 1000)
        self.ic_update_interval = cfg.get("ic_update_interval_hours", 24)
        self.next_ic_update = datetime.now(UTC)

    def initial_sync(self):
        """启动时检查并更新所有关键数据，然后生成一次交易信号"""
        self.symbols = self.dl.get_top_symbols(50)
        intervals = ["5m", "15m", "1h", "4h", "d1"]
        for iv in intervals:
            self.safe_call(self.update_klines, self.symbols, iv)
        self.safe_call(self.update_oi_and_order_book, self.symbols)
        # 启动时同步资金费率
        self.safe_call(self.update_funding_rates, self.symbols)
        self.safe_call(self.update_daily_data, self.symbols)
        self.safe_call(self.update_ic_scores_from_db)
        self.safe_call(self.generate_signals, self.symbols)
        self.next_ic_update = self._calc_next_ic_update(datetime.now(UTC))
        
    def safe_call(self, func, *args, **kwargs):
        """Execute func with error logging."""
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("%s failed: %s", func.__name__, e)

    def update_klines(self, symbols, interval):
        for sym in symbols:
            try:
                self.dl.incremental_update_klines(sym, interval)
            except Exception as e:
                logging.exception("update %s %s failed: %s", sym, interval, e)

    def update_oi_and_order_book(self, symbols):
        for sym in symbols:
            try:
                self.dl.update_open_interest(sym)
            except Exception as e:
                logging.exception("update open interest %s failed: %s", sym, e)
            try:
                self.dl.update_order_book(sym)
            except Exception as e:
                logging.exception("update order book %s failed: %s", sym, e)

    def update_funding_rates(self, symbols):
        for sym in symbols:
            try:
                self.dl.update_funding_rate(sym)
            except Exception as e:
                logging.exception("update funding rate %s failed: %s", sym, e)

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

    def update_features(self):
        """重新计算特征并写入数据库"""
        try:
            symbols = self.dl.get_top_symbols(self.fe.topn)
            intervals = ["5m", "15m", "1h", "4h", "d1"]
            for iv in intervals:
                self.update_klines(symbols, iv)
            self.update_oi_and_order_book(symbols)
            self.update_funding_rates(symbols)
            self.fe.merge_features(topn=self.fe.topn, save_to_db=True, batch_size=1)
        except Exception as e:
            logging.exception("update_features failed: %s", e)

    def update_ic_scores_from_db(self):
        """Load recent features and update factor IC scores."""
        try:
            query = text(
                "SELECT * FROM features ORDER BY open_time DESC LIMIT :n"
            )
            df = pd.read_sql(
                query,
                self.engine,
                params={"n": self.ic_update_limit},
                parse_dates=["open_time", "close_time"],
            )
            if df.empty:
                logging.warning("update_ic_scores_from_db: no data returned")
                return
            df = df.sort_values("open_time")
            self.sg.update_ic_scores(df)
            logging.info("[update_ic_scores] %s", self.sg.current_weights)
        except Exception as e:
            logging.exception("update_ic_scores_from_db failed: %s", e)

    def _calc_next_ic_update(self, now):
        if self.ic_update_interval == 24:
            return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (now + timedelta(hours=self.ic_update_interval)).replace(minute=0, second=0, microsecond=0)

    def generate_signals(self, symbols):
        logging.info("generating signals for %s symbols", len(symbols))
        global_metrics = load_global_metrics(self.engine)
        results = []
        now = datetime.now(UTC).replace(second=0, microsecond=0)
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
                raw1h = {k: _to_builtin(v) for k, v in raw1h.items()}
                raw4h = {k: _to_builtin(v) for k, v in raw4h.items()}
                rawd1 = {k: _to_builtin(v) for k, v in rawd1.items()}
                data = {
                    "symbol": sym,
                    "time": now,
                    "price": feats1h.get("close"),
                    "signal": sig.get("signal"),
                    "score": sig.get("score"),
                    "pos": sig.get("position_size"),
                    "take_profit": sig.get("take_profit"),
                    "stop_loss": sig.get("stop_loss"),
                    "indicators": safe_json_dumps(
                        {
                            "feat_1h": raw1h,
                            "feat_4h": raw4h,
                            "feat_d1": rawd1,
                            "details": sig.get("details"),
                        },
                    ),
                }
                data = {k: _to_builtin(v) for k, v in data.items()}
                results.append(data)
            except Exception as e:
                logging.exception("signal for %s failed: %s", sym, e)
        if not results:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO live_full_data "
                    "(`symbol`,`time`,`price`,`signal`,`score`,`pos`,`take_profit`,`stop_loss`,`indicators`) "
                    "VALUES (:symbol,:time,:price,:signal,:score,:pos,:take_profit,:stop_loss,:indicators)"
                ),
                results,
            )
        logging.info("[generate_signals] wrote %s rows to live_full_data", len(results))
        # filter out entries without a trading signal before ranking
        filtered = [r for r in results if r.get("signal")]
        filtered.sort(key=lambda x: abs(x.get("score") or 0), reverse=True)
        top10 = filtered[:10]
        if top10:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "REPLACE INTO live_top10_signals "
                        "(`symbol`,`time`,`price`,`signal`,`score`,`pos`,`take_profit`,`stop_loss`,`indicators`) "
                        "VALUES (:symbol,:time,:price,:signal,:score,:pos,:take_profit,:stop_loss,:indicators)"
                    ),
                    top10,
                )
            logging.info(
                "[generate_signals] wrote %s rows to live_top10_signals", len(top10)
            )

    def schedule_next(self):
        next_run = (datetime.now(UTC) + timedelta(minutes=1)).replace(
            second=0, microsecond=0
        )
        self.scheduler.enterabs(next_run.timestamp(), 1, self.dispatch_tasks)

    def dispatch_tasks(self):
        now = datetime.now(UTC)
        if now >= self.next_symbols_refresh:
            self.symbols = self.dl.get_top_symbols()
            self.next_symbols_refresh = now + timedelta(hours=1)

        minute = now.minute
        tasks = []
        if minute % 5 == 0 and minute != 0:
            tasks.append(
                self.executor.submit(
                    self.safe_call, self.update_oi_and_order_book, self.symbols
                )
            )
        if minute % 15 == 0 and minute != 0:
            tasks.append(
                self.executor.submit(self.safe_call, self.update_klines, self.symbols, "5m")
            )
        if minute % 30 == 0 and minute != 0:
            tasks.append(
                self.executor.submit(self.safe_call, self.update_klines, self.symbols, "15m")
            )
        if now >= self.next_ic_update:
            tasks.append(
                self.executor.submit(
                    self.safe_call, self.update_ic_scores_from_db
                )
            )
            self.next_ic_update = self._calc_next_ic_update(now)

        if minute == 0:
            tasks.append(
                self.executor.submit(
                    self.safe_call, self.update_oi_and_order_book, self.symbols
                )
            )
            update_future = self.executor.submit(
                self.safe_call, self.update_klines, self.symbols, "1h"
            )
            tasks.append(update_future)
            if now.hour % 4 == 0:
                tasks.append(
                    self.executor.submit(
                        self.safe_call, self.update_klines, self.symbols, "4h"
                    )
                )
            if now.hour % 8 == 0:
                tasks.append(
                    self.executor.submit(
                        self.safe_call, self.update_funding_rates, self.symbols
                    )
                )
            if now.hour == 0:
                self.safe_call(self.update_klines, self.symbols, "d1")
                self.safe_call(self.update_daily_data, self.symbols)
                self.safe_call(self.update_features)
            update_future.result()
            tasks.append(
                self.executor.submit(
                    self.safe_call, self.generate_signals, self.symbols
                )
            )

        for f in tasks:
            try:
                f.result()
            except Exception as e:
                logging.exception("task failed: %s", e)

        self.schedule_next()

    def run(self):
        self.initial_sync()
        self.schedule_next()
        while True:
            self.scheduler.run(blocking=False)
            if self.scheduler.queue:
                next_time = self.scheduler.queue[0].time
                sleep_secs = next_time - time.time()
                time.sleep(max(sleep_secs, 0))
            else:
                time.sleep(1)


if __name__ == "__main__":
    Scheduler().run()

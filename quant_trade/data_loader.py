# -*- coding: utf-8 -*-
"""
DataLoader v2.3-patch1   (2025-06-03)
==================================================================
同步包含情绪、fundingRate、K 线。
"""

from __future__ import annotations
import os, time, re, logging, threading, datetime as dt
from typing import Dict, List, Optional

if __package__ is None and __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "quant_trade"

import json
import yaml, requests, pandas as pd, numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.exc import IntegrityError
from quant_trade.utils.ratelimiter import RateLimiter  # 你的限速器
from quant_trade.utils.helper import calc_order_book_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _safe_retry(fn, retries: int = 3, backoff: float = 1.0,
                retry_on: tuple = (requests.exceptions.RequestException, BinanceAPIException)):
    """简单指数退避重试包装

    若捕获到 BinanceAPIException 且 code 为 -1003，则额外延长等待时间，
    以避免持续触发限速错误。
    """
    last_exc = None
    for i in range(retries):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            delay = backoff * (2 ** i)
            if isinstance(exc, BinanceAPIException) and getattr(exc, "code", None) == -1003:
                delay *= 5
                logger.warning("Hit Binance rate limit (-1003), sleep %.1fs", delay)
            else:
                logger.warning("Retry %s/%s for %s: %s", i + 1, retries, fn.__qualname__, exc)
            time.sleep(delay)
    raise RuntimeError(
        f"Failed after {retries} retries for {fn.__qualname__}"
    ) from last_exc


def compute_vix_proxy(funding_rate: Optional[float], oi_chg: Optional[float]) -> Optional[float]:
    """根据资金费率与持仓量变化计算简易波动率代理"""
    if funding_rate is None or oi_chg is None:
        return None
    try:
        return 0.5 * float(funding_rate) + 0.5 * float(oi_chg)
    except Exception:
        return None


class DataLoader:

    _sentiment_cache: Optional[pd.DataFrame] = None
    _funding_cache: Dict[str, pd.DataFrame] = {}
    _cg_market_cache: Dict[str, pd.DataFrame] = {}
    _cache_lock = threading.Lock()





    def __init__(self, config_path: str = "utils/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Binance client
        bin_cfg = cfg.get("binance", {})
        self.client = Client(
            api_key=os.getenv("BINANCE_API_KEY", bin_cfg.get("api_key", "")),
            api_secret=os.getenv("BINANCE_API_SECRET", bin_cfg.get("api_secret", "")),
        )
        if bin_cfg.get("proxy"):
            self.client.session.proxies.update(bin_cfg["proxy"])
        self.kl_rate_limiter = RateLimiter(max_calls=5, period=1.0)

        # CoinGecko client (only requests with API key)
        cg_cfg = cfg.get("coingecko", {})
        self.cg_api_key = os.getenv("COINGECKO_API_KEY", cg_cfg.get("api_key", ""))
        self.cg_rate_limiter = RateLimiter(max_calls=30, period=60)
        self.cg_id_file = os.path.join(os.path.dirname(__file__), "coingecko_ids.json")
        if os.path.exists(self.cg_id_file):
            try:
                with open(self.cg_id_file, "r", encoding="utf-8") as f:
                    self._cg_id_map = json.load(f)
            except Exception as e:
                logger.warning("[coingecko] load id map fail: %s", e)
                self._cg_id_map = {}
        else:
            self._cg_id_map: Dict[str, str] = {}

        # MySQL
        mysql_cfg = cfg["mysql"]
        conn_str = (
            f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
            f"@{mysql_cfg['host']}:{mysql_cfg.get('port',3306)}/{mysql_cfg['database']}"
            f"?charset={mysql_cfg.get('charset','utf8mb4')}"
        )
        self.engine = create_engine(conn_str, pool_recycle=3600)

        # CoinMetrics loader
        from .coinmetrics_loader import CoinMetricsLoader

        cm_cfg = cfg.get("coinmetrics", {})
        cm_metrics = cm_cfg.get("metrics") or []
        self.cm_loader = CoinMetricsLoader(
            self.engine,
            api_key=os.getenv("COINMETRICS_API_KEY", cm_cfg.get("api_key", "")),
            metrics=cm_metrics,
        )

        # Params
        dl = cfg.get("data_loader", {})
        self.topn       = dl.get("topn", 20)
        self.main_iv    = dl.get("interval", "4h")
        self.aux_ivs    = dl.get("aux_interval", ["1h"])
        if isinstance(self.aux_ivs, str):
            self.aux_ivs = [self.aux_ivs]
        start_cfg = dl.get("start")
        if start_cfg in (None, "", "auto-1y"):
            self.since = (dt.date.today() - dt.timedelta(days=365)).isoformat()
        else:
            self.since = str(start_cfg)
        self.till       = dl.get("end")
        self.retries    = dl.get("retries", 3)
        self.backoff    = dl.get("backoff", 1.0)
        self.excluded   = dl.get("excluded_list", [])

    # ───────────────────────────── FG 指数 ────────────────────────────────
    SENTIMENT_URL = "https://api.alternative.me/fng/?limit=0&format=json"

    def update_sentiment(self) -> None:
        rows = _safe_retry(lambda: requests.get(self.SENTIMENT_URL, timeout=10).json(),
                           retries=self.retries, backoff=self.backoff).get("data", [])
        if not rows:
            logger.warning("[sentiment] API empty")
            return
        df = pd.DataFrame(rows)
        df["fg_index"] = df["value"].astype(float) / 100.0
        df = df[["timestamp", "value", "value_classification", "fg_index"]].astype(
            {"timestamp": "int", "value": "str", "value_classification": "str"}
        )
        sql = (
            "REPLACE INTO sentiment (timestamp,value,value_classification,fg_index) "
            "VALUES (:timestamp,:value,:value_classification,:fg_index)"
        )
        with self.engine.begin() as conn:
            conn.execute(text(sql), df.to_dict("records"))
        logger.info("[sentiment] %s rows", len(df))
        with self._cache_lock:
            self._sentiment_cache = None

    def _sentiment_df(self) -> pd.DataFrame:
        with self._cache_lock:
            if self._sentiment_cache is None:
                q = "SELECT timestamp AS open_time, fg_index FROM sentiment ORDER BY timestamp"
                self._sentiment_cache = pd.read_sql(q, self.engine, parse_dates=["open_time"])
            return self._sentiment_cache

    # ───────────────────────────── 社交情绪 ─────────────────────────────
    def update_social_sentiment(self) -> None:
        """Fetch social sentiment scores and store into database."""
        from .social_sentiment_loader import SocialSentimentLoader

        last = pd.read_sql(
            "SELECT MAX(date) AS d FROM social_sentiment",
            self.engine,
            parse_dates=["d"],
        )
        since = (last["d"].iloc[0].date() + dt.timedelta(days=1)) if not last.empty and pd.notnull(last["d"].iloc[0]) else dt.date.today() - dt.timedelta(days=7)
        loader = SocialSentimentLoader(self.engine, retries=self.retries, backoff=self.backoff)
        loader.update_scores(since)

    # ───────────────────────────── Funding Rate ──────────────────────────
    FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

    def update_funding_rate(self, symbol: str) -> None:
        start_dt = pd.to_datetime(self.since)
        last = pd.read_sql(
            text("SELECT fundingTime FROM funding_rate WHERE symbol=:s ORDER BY fundingTime DESC LIMIT 1"),
            self.engine, params={"s": symbol}, parse_dates=["fundingTime"]
        )
        if not last.empty:
            start_dt = last["fundingTime"].iloc[0]
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        rows: List[dict] = []
        while start_ms < end_ms:
            params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
            payload = _safe_retry(
                lambda: requests.get(self.FUNDING_URL, params=params, timeout=10).json(),
                retries=self.retries, backoff=self.backoff
            )
            if not payload:
                break
            rows.extend(payload)
            start_ms = payload[-1]["fundingTime"] + 1
            time.sleep(0.2)
        if not rows:
            return
        df = pd.DataFrame(rows)[["fundingTime", "fundingRate"]]
        df.insert(0, "symbol", symbol)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        with self.engine.begin() as conn:
            conn.execute(text(
                "REPLACE INTO funding_rate (symbol, fundingTime, fundingRate) "
                "VALUES (:symbol,:fundingTime,:fundingRate)"
            ), df.to_dict("records"))
        logger.info("[funding] %s ↦ %s", symbol, len(df))

    def _funding_df(self, symbol: str) -> pd.DataFrame:
        with self._cache_lock:
            if symbol not in self._funding_cache:
                q = (
                    "SELECT fundingTime AS open_time, fundingRate AS funding_rate "
                    "FROM funding_rate WHERE symbol=:sym ORDER BY fundingTime"
                )
                self._funding_cache[symbol] = pd.read_sql(text(q), self.engine,
                                                          params={"sym": symbol},
                                                          parse_dates=["open_time"])
            return self._funding_cache[symbol]

    def _cg_market_df(self, symbol: str) -> pd.DataFrame:
        """读取 CoinGecko 市值数据并缓存"""
        with self._cache_lock:
            if symbol not in self._cg_market_cache:
                q = (
                    "SELECT timestamp AS open_time, price AS cg_price, "
                    "market_cap AS cg_market_cap, total_volume AS cg_total_volume "
                    "FROM cg_market_data WHERE symbol=:sym ORDER BY timestamp"
                )
                self._cg_market_cache[symbol] = pd.read_sql(
                    text(q), self.engine, params={"sym": symbol}, parse_dates=["open_time"]
                )
            return self._cg_market_cache[symbol]

    # ───────────────────────────── Open Interest ─────────────────────────
    def update_open_interest(self, symbol: str) -> None:
        """同步单个合约的当前持仓量"""
        payload = _safe_retry(
            lambda: self.client.futures_open_interest(symbol=symbol),
            retries=self.retries,
            backoff=self.backoff,
        )
        if not payload:
            return
        ts = pd.to_datetime(payload.get("time", int(time.time() * 1000)), unit="ms")
        oi = float(payload.get("openInterest", 0))
        df = pd.DataFrame([
            {"symbol": symbol, "timestamp": ts, "open_interest": oi}
        ])
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO open_interest (symbol, timestamp, open_interest) "
                    "VALUES (:symbol,:timestamp,:open_interest)"
                ),
                df.to_dict("records"),
            )
        logger.info("[open_interest] %s", symbol)

    def get_latest_open_interest(self, symbol: str) -> Optional[dict]:
        """返回指定合约最近两条持仓量及变化率"""
        q = (
            "SELECT timestamp, open_interest FROM open_interest "
            "WHERE symbol=:s ORDER BY timestamp DESC LIMIT 2"
        )
        df = pd.read_sql(text(q), self.engine, params={"s": symbol}, parse_dates=["timestamp"])
        if df.empty:
            return None
        latest = df.iloc[0]
        if len(df) > 1 and df.iloc[1]["open_interest"]:
            prev = df.iloc[1]
            prev_val = prev["open_interest"]
            if prev_val:
                oi_chg = (latest["open_interest"] - prev_val) / prev_val
            else:
                oi_chg = None
        else:
            oi_chg = None
        fr_q = (
            "SELECT fundingRate FROM funding_rate "
            "WHERE symbol=:s ORDER BY fundingTime DESC LIMIT 1"
        )
        fr_df = pd.read_sql(text(fr_q), self.engine, params={"s": symbol})
        funding_rate = float(fr_df["fundingRate"].iloc[0]) if not fr_df.empty else None
        vix_p = compute_vix_proxy(funding_rate, oi_chg)
        return {
            "timestamp": latest["timestamp"],
            "open_interest": float(latest["open_interest"]),
            "oi_chg": float(oi_chg) if oi_chg is not None else None,
            "vix_proxy": float(vix_p) if vix_p is not None else None,
        }

    # ───────────────────────────── Order Book ────────────────────────────
    def update_order_book(self, symbol: str) -> None:
        """抓取并保存深度快照 (前10档)"""
        payload = _safe_retry(
            lambda: self.client.futures_order_book(symbol=symbol, limit=10),
            retries=self.retries,
            backoff=self.backoff,
        )
        if not payload:
            return
        ts = pd.Timestamp.utcnow().tz_localize(None)
        bids = json.dumps(payload.get("bids", []))
        asks = json.dumps(payload.get("asks", []))
        df = pd.DataFrame([
            {"symbol": symbol, "timestamp": ts, "bids": bids, "asks": asks}
        ])
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO order_book (symbol, timestamp, bids, asks) "
                    "VALUES (:symbol,:timestamp,:bids,:asks)"
                ),
                df.to_dict("records"),
            )
        logger.info("[order_book] %s", symbol)

    def get_latest_order_book_imbalance(self, symbol: str) -> Optional[float]:
        """返回指定合约最近一次盘口失衡值"""
        q = (
            "SELECT timestamp, bids, asks FROM order_book "
            "WHERE symbol=:s ORDER BY timestamp DESC LIMIT 1"
        )
        df = pd.read_sql(text(q), self.engine, params={"s": symbol})
        if df.empty:
            return None
        feats = calc_order_book_features(df)
        val = feats["bid_ask_imbalance"].iloc[0]
        return float(val) if pd.notnull(val) else None


    # ───────────────────────────── CoinGecko 辅助数据 ──────────────────────
    CG_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
    CG_MARKET_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
    CG_MARKET_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
    CG_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
    CG_COIN_INFO_URL = "https://api.coingecko.com/api/v3/coins/{id}"
    CG_CATEGORIES_URL = "https://api.coingecko.com/api/v3/coins/categories"

    def _cg_headers(self) -> Dict[str, str]:
        """返回访问 CoinGecko API 所需的请求头"""
        return {"x-cg-demo-api-key": self.cg_api_key} if self.cg_api_key else {}

    def _cg_get_id(self, symbol: str) -> Optional[str]:
        if symbol in self._cg_id_map:
            return self._cg_id_map[symbol]
        base = re.sub("USDT$", "", symbol).lower()
        self.cg_rate_limiter.acquire()
        res = _safe_retry(
            lambda: requests.get(self.CG_SEARCH_URL, params={"query": base}, headers=self._cg_headers(), timeout=10).json(),
            retries=self.retries, backoff=self.backoff
        )
        for coin in res.get("coins", []):
            if coin.get("symbol", "").lower() == base:
                cid = coin["id"]
                self._cg_id_map[symbol] = cid
                try:
                    with open(self.cg_id_file, "w", encoding="utf-8") as f:
                        json.dump(self._cg_id_map, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning("[coingecko] save id map fail: %s", e)
                return cid
        if res.get("coins"):
            cid = res["coins"][0]["id"]
            self._cg_id_map[symbol] = cid
            try:
                with open(self.cg_id_file, "w", encoding="utf-8") as f:
                    json.dump(self._cg_id_map, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("[coingecko] save id map fail: %s", e)
            return cid
        return None

    def update_cg_market_data(self, symbols: List[str]) -> None:
        """拉取 CoinGecko 市值信息，自动回补缺失区间"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()
        headers = self._cg_headers()
        rows: List[Dict[str, object]] = []

        today = pd.Timestamp.utcnow().floor("D").tz_localize(None)
        tomorrow = today + dt.timedelta(days=365)

        stmt = text(
            "SELECT symbol, MAX(timestamp) AS ts FROM cg_market_data "
            "WHERE symbol IN :symbols GROUP BY symbol"
        ).bindparams(bindparam("symbols", expanding=True))
        last_df = pd.read_sql(stmt, self.engine, params={"symbols": symbols}, parse_dates=["ts"])
        last_map = {r["symbol"]: r["ts"] for _, r in last_df.iterrows()}

        def fetch_symbol(sym: str) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            last_ts = last_map.get(sym)
            if last_ts is None or pd.isna(last_ts):
                start = today - dt.timedelta(days=365)
            else:
                start = last_ts + dt.timedelta(days=1)
            if start > today:
                return out
            cid = self._cg_get_id(sym)
            if not cid:
                return out
            self.cg_rate_limiter.acquire()
            data = _safe_retry(
                lambda: requests.get(
                    self.CG_MARKET_RANGE_URL.format(id=cid),
                    params={
                        "vs_currency": "usd",
                        "from": int(start.timestamp()),
                        "to": int(tomorrow.timestamp()),
                    },
                    headers=headers,
                    timeout=10,
                ).json(),
                retries=self.retries,
                backoff=self.backoff,
            )
            prices = data.get("prices", [])
            m_caps = data.get("market_caps", [])
            volumes = data.get("total_volumes", [])
            for p, m, v in zip(prices, m_caps, volumes):
                ts = pd.to_datetime(p[0], unit="ms").to_pydatetime().replace(tzinfo=None)
                out.append({
                    "symbol": sym,
                    "timestamp": ts,
                    "price": float(p[1]),
                    "market_cap": float(m[1]),
                    "total_volume": float(v[1]),
                })
            logger.info("[cg_market] %s fetched %s rows", sym, len(out))
            time.sleep(0.1)
            return out

        logger.info("[cg_market] fetch %s symbols…", len(symbols))
        max_workers = min(4, max(1, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_symbol, s): s for s in symbols}
            for f in as_completed(futures):
                try:
                    rows.extend(f.result())
                except Exception as e:  # pragma: no cover - unexpected errors
                    logger.exception("[cg_market] worker err: %s", e)

        if not rows:
            logger.info("[cg_market] no new rows")
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO cg_market_data (symbol, timestamp, price, market_cap, total_volume) "
                    "VALUES (:symbol, :timestamp, :price, :market_cap, :total_volume)"
                ),
                rows,
            )
        elapsed = time.time() - start_time
        logger.info("[cg_market] 写入 %s 行 (%.2fs)", len(rows), elapsed)

    def update_cg_global_metrics(self, min_interval_hours: float = 24.0) -> None:
        """更新 CoinGecko 全局指标
        :param min_interval_hours: 与上次更新时间相隔的日度间隔(小时)。默认按日刷新，
                                   即仅在 UTC 零点后才会重新获取数据。
        """
        now_hour = pd.Timestamp.utcnow().floor("h").tz_localize(None)

        if min_interval_hours > 0:
            df_last = pd.read_sql(
                "SELECT timestamp FROM cg_global_metrics ORDER BY timestamp DESC LIMIT 1",
                self.engine,
                parse_dates=["timestamp"],
            )
            if not df_last.empty:
                last_ts = df_last.iloc[0]["timestamp"]
                if last_ts is not None:
                    if last_ts.tzinfo is not None:
                        last_ts = last_ts.tz_convert("UTC").tz_localize(None)
                    last_day = last_ts.floor("D")
                    next_allowed = last_day + pd.Timedelta(hours=min_interval_hours)
                    if now_hour < next_allowed:
                        logger.info(
                            "[cg_global] skip update (<%sh since %s)",
                            min_interval_hours,
                            last_ts,
                        )
                        return
        headers = self._cg_headers()
        self.cg_rate_limiter.acquire()
        data = _safe_retry(
            lambda: requests.get(self.CG_GLOBAL_URL, headers=headers, timeout=10).json(),
            retries=self.retries,
            backoff=self.backoff,
        ).get("data", {})
        if not data:
            return
        row = {
            "timestamp": now_hour.to_pydatetime().replace(tzinfo=None),
            "total_market_cap": data.get("total_market_cap", {}).get("usd"),
            "total_volume": data.get("total_volume", {}).get("usd"),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
        }
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO cg_global_metrics (timestamp, total_market_cap, total_volume, btc_dominance, eth_dominance) "
                    "VALUES (:timestamp, :total_market_cap, :total_volume, :btc_dominance, :eth_dominance)"
                ),
                [row],
            )
        logger.info("[cg_global] updated")

    def update_cg_coin_categories(self, symbols: List[str]) -> None:
        """按日更新币种所属的板块信息，避免过度调用"""
        headers = self._cg_headers()
        today = dt.date.today()

        stmt = text(
            "SELECT symbol, last_updated FROM cg_coin_categories WHERE symbol IN :syms"
        ).bindparams(bindparam("syms", expanding=True))
        last_df = pd.read_sql(stmt, self.engine, params={"syms": symbols}, parse_dates=["last_updated"])
        last_map = {r["symbol"]: r["last_updated"].date() if r["last_updated"] is not None else None for _, r in last_df.iterrows()}

        rows = []
        for sym in symbols:
            if last_map.get(sym) == today:
                continue
            cid = self._cg_get_id(sym)
            if not cid:
                continue
            self.cg_rate_limiter.acquire()
            data = _safe_retry(
                lambda: requests.get(
                    self.CG_COIN_INFO_URL.format(id=cid),
                    params={
                        "localization": "false",
                        "tickers": "false",
                        "market_data": "false",
                        "community_data": "false",
                        "developer_data": "false",
                        "sparkline": "false",
                    },
                    headers=headers,
                    timeout=10,
                ).json(),
                retries=self.retries,
                backoff=self.backoff,
            )
            cats = data.get("categories", [])
            rows.append({
                "symbol": sym,
                "categories": ",".join(cats),
                "last_updated": today.isoformat(),
            })
            time.sleep(0.1)

        if rows:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "REPLACE INTO cg_coin_categories (symbol, categories, last_updated) "
                        "VALUES (:symbol, :categories, :last_updated)"
                    ),
                    rows,
                )
            logger.info("[cg_categories] %s rows", len(rows))

    def update_cg_category_stats(self) -> None:
        """获取 CoinGecko 各板块市值等信息并保存，按日刷新"""
        today = dt.date.today()
        df_last = pd.read_sql(
            "SELECT MAX(updated_at) AS ts FROM cg_category_stats",
            self.engine,
            parse_dates=["ts"],
        )
        if not df_last.empty and df_last.iloc[0]["ts"] is not None:
            if df_last.iloc[0]["ts"].date() == today:
                logger.info("[cg_category_stats] skip update (already today)")
                return

        headers = self._cg_headers()
        self.cg_rate_limiter.acquire()
        data = _safe_retry(
            lambda: requests.get(self.CG_CATEGORIES_URL, headers=headers, timeout=10).json(),
            retries=self.retries,
            backoff=self.backoff,
        )
        if not isinstance(data, list):
            return
        rows = []
        for r in data:
            ts = r.get("updated_at")
            if ts:
                ts = pd.to_datetime(ts).to_pydatetime().replace(tzinfo=None)
            rows.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "market_cap": r.get("market_cap"),
                "market_cap_change_24h": r.get("market_cap_change_24h"),
                "volume_24h": r.get("volume_24h"),
                "top_3_coins": ",".join(r.get("top_3_coins", [])),
                "updated_at": ts,
            })
        if rows:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "REPLACE INTO cg_category_stats (id, name, market_cap, market_cap_change_24h, volume_24h, top_3_coins, updated_at) "
                        "VALUES (:id, :name, :market_cap, :market_cap_change_24h, :volume_24h, :top_3_coins, :updated_at)"
                    ),
                    rows,
                )
            logger.info("[cg_category_stats] %s rows", len(rows))

    def update_cm_metrics(self, symbols: List[str]) -> None:
        """更新链上指标"""
        if not self.cm_loader.metrics:
            return
        self.cm_loader.update_cm_metrics(symbols)

    def get_hot_sector(self) -> Optional[dict]:
        """根据最新的 volume_24h 判断热门板块"""
        q = (
            "SELECT id, name, market_cap, market_cap_change_24h, volume_24h, top_3_coins "
            "FROM cg_category_stats "
            "WHERE updated_at = (SELECT MAX(updated_at) FROM cg_category_stats)"
        )
        df = pd.read_sql(q, self.engine)
        df = df.dropna(subset=["name", "volume_24h"])
        if df.empty:
            return None
        df["volume_24h"] = pd.to_numeric(df["volume_24h"], errors="coerce")
        df = df.dropna(subset=["volume_24h"])
        if df.empty:
            return None
        total = df["volume_24h"].sum()
        df = df.sort_values("volume_24h", ascending=False)
        top = df.iloc[0]
        strength = float(top["volume_24h"]) / total if total else None

        return {"hot_sector": top["name"], "hot_sector_strength": strength}

    def get_latest_cg_global_metrics(self, symbol: Optional[str] = None) -> Optional[dict]:
        """返回最近两条 CoinGecko 全球指标并附带变化率，可选给出指定币种的板块相关性"""
        q = (
            "SELECT timestamp, total_market_cap, total_volume, btc_dominance, eth_dominance "
            "FROM cg_global_metrics ORDER BY timestamp DESC LIMIT 2"
        )
        df = pd.read_sql(q, self.engine, parse_dates=["timestamp"])
        if df.empty:
            return None
        latest = df.iloc[0]
        if len(df) > 1:
            prev = df.iloc[1]
            def pct(cur, prev_val):
                return (cur - prev_val) / prev_val if prev_val else None
            btc_dom_chg = pct(latest["btc_dominance"], prev["btc_dominance"])
            mcap_growth = pct(latest["total_market_cap"], prev["total_market_cap"])
            vol_chg = pct(latest["total_volume"], prev["total_volume"])
        else:
            btc_dom_chg = mcap_growth = vol_chg = None
        metrics = {
            "timestamp": latest["timestamp"],
            "btc_dom_chg": float(btc_dom_chg) if btc_dom_chg is not None else None,
            "mcap_growth": float(mcap_growth) if mcap_growth is not None else None,
            "vol_chg": float(vol_chg) if vol_chg is not None else None,
            "btc_dominance": float(latest["btc_dominance"]),
            "total_market_cap": float(latest["total_market_cap"]),
            "total_volume": float(latest["total_volume"]),
            "eth_dominance": float(latest["eth_dominance"]),
        }
        oi = self.get_latest_open_interest("BTCUSDT")
        if oi and oi.get("vix_proxy") is not None:
            metrics["vix_proxy"] = oi["vix_proxy"]
        hot = self.get_hot_sector()
        if hot:
            metrics.update(hot)
            if symbol is not None:
                df_cat = pd.read_sql(
                    text("SELECT categories FROM cg_coin_categories WHERE symbol=:s"),
                    self.engine,
                    params={"s": symbol},
                )
                if not df_cat.empty:
                    cats = df_cat["categories"].iloc[0]
                    if isinstance(cats, str) and cats:
                        arr = [c.strip() for c in cats.split(",") if c.strip()]
                        metrics["sector_corr"] = (
                            1.0 if hot["hot_sector"] in arr else 0.0
                        )
        return metrics

    # ───────────────────────────── 选币逻辑 ───────────────────────────────
    def get_top_symbols(self, n: Optional[int] = None) -> List[str]:
        # 1. 拉所有“处于TRADING状态”的永续USDT合约
        info = _safe_retry(
            lambda: self.client.futures_exchange_info(),
            retries=self.retries,
            backoff=self.backoff,
        )
        trading_perp_usdt = {
            s['symbol'] for s in info['symbols']
            if s['status'] == 'TRADING'
               and s['contractType'] == 'PERPETUAL'
               and s['quoteAsset'] == 'USDT'
        }
        # 2. 继续用24小时成交量筛主流币
        tickers = _safe_retry(lambda: self.client.futures_ticker(),
                              retries=self.retries, backoff=self.backoff)
        cands = [
            (t["symbol"], float(t["quoteVolume"]))
            for t in tickers
            if t["symbol"] in trading_perp_usdt
               and t["symbol"] not in self.excluded
               and not t["symbol"].startswith("1000")
               and float(t["quoteVolume"]) > 0
        ]
        cands.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in cands[: n or self.topn]]
        logger.info("[symbol-select] %s", top)
        return top
    # ───────────────────────────── KLine 同步 ────────────────────────────
    def fetch_klines_raw(self, symbol: str, interval: str,
                         start: pd.Timestamp, end: pd.Timestamp) -> list:
        """
        安全获取K线：只有币安返回-1121才raise BinanceAPIException，其它情况只log
        """
        self.kl_rate_limiter.acquire()  # 你原有的限速器，继续用
        api_iv = "1d" if interval == "d1" else interval
        try:
            res = self.client.futures_klines(
                symbol=symbol,
                interval=api_iv,
                startTime=int(start.timestamp() * 1000),
                endTime=int(end.timestamp() * 1000),
                limit=1000,
            )
        except Exception as e:
            logger.warning(f"[{symbol}] K线API请求直接抛异常: {e}")
            raise

        if isinstance(res, dict):
            code = res.get("code", -1)
            msg = res.get("msg", "Binance返回dict")
            if code == -1121:
                raise BinanceAPIException(None, code, msg)  # 只排除-1121
            logger.warning(f"[{symbol}] K线API异常(code={code}): {msg}，但不是-1121，不排除")
            return []
        return res

    def incremental_update_klines(self, symbol: str, interval: str) -> None:
        """
        增量同步K线。只对-1121才加入排除，其它所有异常仅跳出本币本轮，不排除。
        """
        sql = (
            "SELECT open_time FROM klines "
            "WHERE symbol=:s AND `interval`=:iv "
            "ORDER BY open_time DESC LIMIT 1"
        )
        last = pd.read_sql(text(sql), self.engine,
                           params={"s": symbol, "iv": interval},
                           parse_dates=["open_time"])
        start_dt = last["open_time"].iloc[0] if not last.empty else pd.to_datetime(self.since)
        end_dt = pd.Timestamp.utcnow().tz_localize(None)

        rows = []
        cur = start_dt
        while cur < end_dt:
            try:
                chunk = _safe_retry(
                    lambda: self.fetch_klines_raw(symbol, interval, cur, end_dt),
                    retries=self.retries, backoff=self.backoff
                )
            except BinanceAPIException as e:
                if e.code == -1121:  # 只排除code==-1121
                    logger.warning("[%s] K线接口缺失 → 已排除", symbol)
                    self.excluded.append(symbol)
                    return
                logger.warning(f"[{symbol}] K线API异常: {e}，本轮跳过但不排除")
                return  # 其它异常跳出本币本轮，不排除

            if not chunk:
                break  # 没有数据就跳出，防止死循环
            rows.extend(chunk)
            cur = pd.to_datetime(chunk[-1][0] + 1, unit="ms")
            time.sleep(0.25)  # 你的默认sleep，也可以换成更短/更长

        if not rows:
            return

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        # 新增：只保留已收盘K线
        now = pd.Timestamp.utcnow().replace(tzinfo=None)
        df = df[df["close_time"] <= now]
        flt_cols = [c for c in cols if c not in ("open_time", "close_time", "num_trades", "ignore")]
        df[flt_cols] = df[flt_cols].apply(pd.to_numeric, errors="coerce")
        df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")
        df.insert(0, "symbol", symbol)
        df.insert(1, "interval", interval)

        # Merge FG 指数
        df = pd.merge_asof(
            df.sort_values("open_time"),
            self._sentiment_df(),
            on="open_time",
            direction="backward"
        )

        # Merge funding rate
        fund = self._funding_df(symbol)
        if not fund.empty:
            df = pd.merge_asof(
                df.sort_values("open_time"),
                fund.sort_values("open_time"),
                on="open_time",
                direction="backward"
            )
        else:
            df["funding_rate"] = None

        # Merge CoinGecko market data
        cg_df = self._cg_market_df(symbol)
        if not cg_df.empty:
            df = pd.merge_asof(
                df.sort_values("open_time"),
                cg_df.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        else:
            df["cg_price"] = None
            df["cg_market_cap"] = None
            df["cg_total_volume"] = None

        # 构建待写入列列表
        cols_final = [
            "symbol", "interval", "open_time", "close_time",
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
            "fg_index", "funding_rate", "cg_price", "cg_market_cap", "cg_total_volume",
        ]

        # NaN → None
        df = df[cols_final].replace({np.nan: None})

        # Drop rows lacking CoinGecko market info
        df = df.dropna(subset=["cg_price"])

        # ==== 加在此处！====
        df = df.dropna(subset=["symbol"])
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df[df["symbol"].str.len() > 0]
        # 如果还不放心再加
        df = df[df["symbol"].str.upper().str.match(r"^[A-Z0-9]+USDT$")]  # 只保留类似 BTCUSDT 这种格式的币种

        # 打印本批唯一 symbol，方便你人工排查异常币名
        # print("本批symbol唯一值:", df["symbol"].unique())
        # ==== 新增：drop掉主K线字段全为0的无效行 ====
        main_cols = ["open", "high", "low", "close", "volume"]
        # 类型安全转换，防止"0"字符串混入
        for col in main_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        before = len(df)
        df = df[~((df[main_cols] == 0).all(axis=1) | df[main_cols].isna().all(axis=1))]
        # 新增更严格过滤
        df = df[df["close"] != 0]
        df = df[df["volume"] != 0]
        after = len(df)
        if before - after > 0:
            logger.info(
                "[%s-%s] K线全为0/NaN的行已剔除：%s 条",
                symbol,
                interval,
                before - after,
            )

        # =============================================
        # 新增空表保护！！！
        if df.empty:
            # print(f"[{symbol}-{interval}] 本批K线全部无效，已跳过写入。")
            return

        # 关键字列名加反引号
        sql_cols = ",".join(f"`{c}`" for c in cols_final)
        sql_vals = ",".join(f":{c}" for c in cols_final)
        sql = f"REPLACE INTO klines ({sql_cols}) VALUES ({sql_vals})"
        with self.engine.begin() as conn:
            conn.execute(text(sql), df.to_dict("records"))
        logger.info("[%s-%s] 写入 %s 行", symbol, interval, len(df))

    # ───────────────────────────── orchestration ───────────────────────────
    def sync_all(self, max_workers: int = 8) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1. 更新 FG 指数
        self.update_sentiment()
        # 1.5 更新 CoinGecko 数据
        symbols = self.get_top_symbols()
        try:
            self.update_cg_global_metrics()
            self.update_cg_market_data(symbols)
            self.update_cg_coin_categories(symbols)
            self.update_cg_category_stats()
        except Exception as e:
            logger.exception("[coingecko] err: %s", e)
        try:
            self.update_cm_metrics(symbols)
        except Exception as e:
            logger.exception("[coinmetrics] err: %s", e)
        # 2. 更新 funding rate / open interest（并发）
        logger.info("[sync] funding/openInterest … (%s)", len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for sym in symbols:
                futures.append(ex.submit(self.update_funding_rate, sym))
                futures.append(ex.submit(self.update_open_interest, sym))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.exception("[funding] worker err: %s", e)

        # 3. 更新 K 线（并发）
        intervals = [self.main_iv] + self.aux_ivs + ["d1"]
        logger.info("[sync] klines … (%s × %s)", len(symbols), intervals)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self.incremental_update_klines, sym, iv)
                for sym in symbols for iv in intervals
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.exception("[klines] worker err: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dl = DataLoader()
    dl.sync_all(max_workers=5)

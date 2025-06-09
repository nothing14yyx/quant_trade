# -*- coding: utf-8 -*-
"""
DataLoader v2.3-patch1 (External indicators removed)  (2025-06-03)
==================================================================
在原版 DataLoader v2.3-patch1 的基础上，去掉所有“外部指标（日线）”相关的方法
（update_onchain_metrics、update_macro_metrics）及其调用，使同步仅包含情绪、fundingRate、K 线。
"""

from __future__ import annotations
import os, time, re, logging, threading, datetime as dt
from typing import Dict, List, Optional

import json
import yaml, requests, pandas as pd, numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from utils.ratelimiter import RateLimiter  # 你的限速器

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _safe_retry(fn, retries: int = 3, backoff: float = 1.0,
                retry_on: tuple = (requests.exceptions.RequestException,)):
    """简单指数退避重试包装"""
    for i in range(retries):
        try:
            return fn()
        except retry_on as exc:
            logger.warning("Retry %s/%s for %s: %s", i + 1, retries, fn.__qualname__, exc)
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed after {retries} retries for {fn.__qualname__}")


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


    # ───────────────────────────── CoinGecko 辅助数据 ──────────────────────
    CG_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
    CG_MARKET_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
    CG_MARKET_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
    CG_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

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
        """仅在缺少当日数据时拉取 CoinGecko 市值信息"""
        headers = self._cg_headers()
        rows = []

        today = pd.Timestamp.utcnow().floor("D").tz_localize(None)
        tomorrow = today + dt.timedelta(days=1)

        for sym in symbols:
            exists = pd.read_sql(
                text(
                    "SELECT 1 FROM cg_market_data WHERE symbol=:sym AND timestamp >= :ts LIMIT 1"
                ),
                self.engine,
                params={"sym": sym, "ts": today},
            )
            if not exists.empty:
                continue
            cid = self._cg_get_id(sym)
            if not cid:
                continue
            self.cg_rate_limiter.acquire()
            data = _safe_retry(
                lambda: requests.get(
                    self.CG_MARKET_RANGE_URL.format(id=cid),
                    params={
                        "vs_currency": "usd",
                        "from": int(today.timestamp()),
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
            if not prices:
                continue
            p, m, v = prices[0], m_caps[0], volumes[0]
            rows.append({
                "symbol": sym,
                "timestamp": today.to_pydatetime(),
                "price": float(p[1]),
                "market_cap": float(m[1]),
                "total_volume": float(v[1]),
            })
            time.sleep(0.1)
        if not rows:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO cg_market_data (symbol, timestamp, price, market_cap, total_volume) "
                    "VALUES (:symbol, :timestamp, :price, :market_cap, :total_volume)"
                ),
                rows,
            )
        logger.info("[cg_market] %s rows", len(rows))

    def update_cg_global_metrics(self) -> None:
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
            "timestamp": pd.Timestamp.utcnow()
                .floor("ms")
                .to_pydatetime()
                .replace(tzinfo=None),
            "total_market_cap": data.get("total_market_cap", {}).get("usd"),
            "total_volume": data.get("total_volume", {}).get("usd"),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
        }
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO cg_global_metrics (timestamp, total_market_cap, total_volume, btc_dominance) "
                    "VALUES (:timestamp, :total_market_cap, :total_volume, :btc_dominance)"
                ),
                [row],
            )
        logger.info("[cg_global] updated")

    # ───────────────────────────── 选币逻辑 ───────────────────────────────
    def get_top_symbols(self, n: Optional[int] = None) -> List[str]:
        # 1. 拉所有“处于TRADING状态”的永续USDT合约
        info = self.client.futures_exchange_info()
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
        try:
            res = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
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
            print(f"[{symbol}-{interval}] K线全为0/NaN的行已剔除：{before - after} 条")

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
        except Exception as e:
            logger.exception("[coingecko] err: %s", e)
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
        intervals = [self.main_iv] + self.aux_ivs + ["1d"]
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

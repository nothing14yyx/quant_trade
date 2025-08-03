# -*- coding: utf-8 -*-
"""CoinMetrics 数据抓取工具"""

from __future__ import annotations

import datetime as dt
import os
import re
import logging
from typing import List, Dict, Optional

import pandas as pd
import requests
from sqlalchemy import text, bindparam

from .data_loader import _safe_retry
from .utils.ratelimiter import RateLimiter

logger = logging.getLogger(__name__)


class CoinMetricsLoader:
    """使用 CoinMetrics Community API 获取链上指标"""

    BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    CATALOG_URL = "https://community-api.coinmetrics.io/v4/catalog/assets"

    def __init__(
        self,
        engine,
        api_key: str = "",
        metrics: Optional[List[str]] = None,
        rate_limit: int = 10,
        period: float = 6.0,
        retries: int = 3,
        backoff: float = 1.0,
    ) -> None:
        self.engine = engine
        self.api_key = api_key or os.getenv("COINMETRICS_API_KEY", "")
        self.metrics = metrics or []
        self.retries = retries
        self.backoff = backoff
        self.rate_limiter = RateLimiter(max_calls=rate_limit, period=period)
        self._metric_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {"X-API-KEY": self.api_key} if self.api_key else {}

    def _asset_code(self, symbol: str) -> str:
        return re.sub("USDT$", "", symbol).lower()

    def community_metrics(self, asset: str) -> List[str]:
        """返回该资产可免费访问的指标列表，结果会缓存"""
        if asset in self._metric_cache:
            return self._metric_cache[asset]

        params = {"assets": asset}
        headers = self._headers()
        data = _safe_retry(
            lambda: requests.get(
                self.CATALOG_URL, params=params, headers=headers, timeout=10
            ).json(),
            retries=self.retries,
            backoff=self.backoff,
        )
        metrics = []
        for item in data.get("data", []):
            for m in item.get("metrics", []):
                freqs = m.get("frequencies", [])
                if any(f.get("community") for f in freqs):
                    metrics.append(m.get("metric"))

        self._metric_cache[asset] = metrics
        return metrics

    def update_cm_metrics(
        self, symbols: List[str], batch_size: int = 10, community_only: bool = False
    ) -> None:
        """按日拉取指定币种的链上指标并保存

        参数:
            symbols: 币种列表
            batch_size: 每次请求包含的指标数, 避免超出 API 限制
        """
        if not symbols or not self.metrics:
            return
        today = dt.date.today().isoformat()
        headers = self._headers()

        stmt = text(
            "SELECT symbol, MAX(timestamp) AS ts FROM cm_onchain_metrics "
            "WHERE symbol IN :syms GROUP BY symbol"
        ).bindparams(bindparam("syms", expanding=True))
        last_df = pd.read_sql(stmt, self.engine, params={"syms": symbols}, parse_dates=["ts"])
        last_map = {r["symbol"]: r["ts"] for _, r in last_df.iterrows()}

        for sym in symbols:
            asset = self._asset_code(sym)
            metrics = self.metrics
            if community_only:
                try:
                    cm_list = self.community_metrics(asset)
                except (requests.exceptions.RequestException, ValueError) as exc:
                    logger.exception("[coinmetrics] %s catalog error: %s", sym, exc)
                    cm_list = []
                metrics = [m for m in metrics if m in cm_list]
            if not metrics:
                continue
            last_ts = last_map.get(sym)
            if last_ts is None or pd.isna(last_ts):
                start = (dt.date.today() - dt.timedelta(days=30)).isoformat()
            else:
                start = (last_ts + dt.timedelta(days=1)).date().isoformat()
            if start > today:
                continue

            rows: List[Dict[str, object]] = []
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i : i + batch_size]
                params = {
                    "assets": asset,
                    "metrics": ",".join(batch),
                    "start_time": start,
                    "end_time": today,
                    "page_size": 1000,
                }
                self.rate_limiter.acquire()
                data = _safe_retry(
                    lambda: requests.get(
                        self.BASE_URL, params=params, headers=headers, timeout=10
                    ).json(),
                    retries=self.retries,
                    backoff=self.backoff,
                )
                if "error" in data or "error_msg" in data:
                    logger.warning(
                        "[coinmetrics] %s error: %s",
                        sym,
                        data.get("error") or data.get("error_msg"),
                    )
                    continue

                found = set()
                for item in data.get("data", []):
                    ts = pd.to_datetime(item["time"]).to_pydatetime().replace(
                        tzinfo=None
                    )
                    for m in batch:
                        val = item.get(m)
                        if val is None:
                            continue
                        try:
                            val = float(val)
                        except (TypeError, ValueError):
                            continue
                        rows.append(
                            {
                                "symbol": sym,
                                "timestamp": ts,
                                "metric": m,
                                "value": val,
                            }
                        )
                        found.add(m)

                missing = set(batch) - found
                if missing:
                    logger.debug(
                        "[coinmetrics] %s metrics not returned: %s",
                        sym,
                        ",".join(sorted(missing)),
                    )
            if rows:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(
                            "REPLACE INTO cm_onchain_metrics (symbol, timestamp, metric, value) "
                            "VALUES (:symbol,:timestamp,:metric,:value)"
                        ),
                        rows,
                    )
                logger.info("[coinmetrics] %s %s rows", sym, len(rows))

__all__ = ["CoinMetricsLoader"]


from __future__ import annotations

import os
import time
import datetime as dt

import pandas as pd
import requests
from sqlalchemy import text

from .data_loader import _safe_retry


class SocialSentimentLoader:
    """基于 CryptoPanic v2 API 抓取新闻情绪并汇总日得分。

    需提供 ``plan`` 与 ``auth_token``，常见查询参数包括 ``kind``、``page``、
    ``public`` 和 ``page_size``。
    """

    API_URL = "https://cryptopanic.com/api/{plan}/v2/posts/"

    def __init__(
        self,
        engine,
        api_key: str = "",
        plan: str = "free",
        retries: int = 3,
        backoff: float = 1.0,
        *,
        public: bool | None = True,
        currencies: str | list[str] | None = None,
        regions: str | list[str] | None = "en",
        filter: str | None = None,
        kind: str | None = "news",
        following: bool = False,
    ) -> None:
        self.engine = engine
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        self.retries = retries
        self.backoff = backoff
        self.API_URL = self.API_URL.format(plan=plan)
        self.public = public
        self.currencies = currencies
        self.regions = regions
        self.filter = filter
        self.kind = kind
        self.following = following

    def _fetch_posts(self, page: int | str = 1) -> dict:
        """Fetch a page of posts, ``page`` may be int or next_url."""
        def _get():
            if isinstance(page, str) and page.startswith("http"):
                r = requests.get(page, timeout=10)
            else:
                params = {
                    "auth_token": self.api_key,
                    "public": str(self.public).lower(),
                    "page_size": 100,
                }
                if self.kind:
                    params["kind"] = self.kind
                if self.currencies:
                    if isinstance(self.currencies, (list, tuple, set)):
                        params["currencies"] = ",".join(self.currencies)
                    else:
                        params["currencies"] = str(self.currencies)

                if self.regions:
                    if isinstance(self.regions, (list, tuple, set)):
                        params["regions"] = ",".join(self.regions)
                    else:
                        params["regions"] = str(self.regions)

                if self.filter:
                    params["filter"] = self.filter

                if self.following:
                    params["following"] = str(self.following).lower()

                if not isinstance(page, str):
                    params["page"] = page
                r = requests.get(self.API_URL, params=params, timeout=10)
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                if r.status_code in (401, 403):
                    raise RuntimeError("Token/套餐无效") from e
                raise
            rem = int(r.headers.get("X-RateLimit-Remaining", "10"))
            if rem < 10:
                time.sleep(60)
            return r.json()

        return _safe_retry(_get, retries=self.retries, backoff=self.backoff)

    def fetch_scores(self, since: dt.date) -> pd.DataFrame:
        """Fetch posts from API and aggregate daily sentiment scores."""
        rows = []
        page: int | str = 1
        while True:
            data = self._fetch_posts(page)
            posts = data.get("data", [])
            if not posts:
                break
            for item in posts:
                ts = pd.to_datetime(item.get("published_at"))
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
                if ts.date() < since:
                    break
                sentiment = str(item.get("sentiment", "")).lower()
                rows.append({"timestamp": ts, "sentiment": sentiment})
            next_url = data.get("next_url")
            if not next_url or ts.date() < since:
                break
            page = next_url

        if not rows:
            return pd.DataFrame(columns=["date", "score"])

        df = pd.DataFrame(rows)
        df["date"] = df["timestamp"].dt.floor("d")
        mapping = {
            "positive": 1.0,
            "bullish": 1.0,
            "mild_bullish": 0.5,
            "negative": -1.0,
            "bearish": -1.0,
            "mild_bearish": -0.5,
            "neutral": 0.0,
        }
        df["score"] = df["sentiment"].map(mapping).fillna(0.0)
        out = df.groupby("date")["score"].mean().reset_index()
        return out

    def update_scores(self, since: dt.date) -> None:
        df = self.fetch_scores(since)
        if df.empty:
            return
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "REPLACE INTO social_sentiment (date, score) VALUES (:date,:score)"
                ),
                df.to_dict("records"),
            )

__all__ = ["SocialSentimentLoader"]

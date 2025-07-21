from __future__ import annotations

import os
import time
import datetime as dt

import pandas as pd
import requests
from sqlalchemy import text

from .data_loader import _safe_retry


class SocialSentimentLoader:
    """Fetch social sentiment data from CryptoPanic or similar API."""

    API_URL = "https://cryptopanic.com/api/free/v2/posts/"

    def __init__(self, engine, api_key: str = "", retries: int = 3, backoff: float = 1.0) -> None:
        self.engine = engine
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        self.retries = retries
        self.backoff = backoff

    def _fetch_posts(self, page: int | str = 1) -> dict:
        """Fetch a page of posts, ``page`` may be int or next_url."""
        def _get():
            if isinstance(page, str) and page.startswith("http"):
                r = requests.get(page, timeout=10)
            else:
                params = {
                    "auth_token": self.api_key,
                    "public": "true",
                }
                if not isinstance(page, str):
                    params["page"] = page
                r = requests.get(self.API_URL, params=params, timeout=10)
            r.raise_for_status()
            rem = int(r.headers.get("X-RateLimit-Remaining", "10"))
            if rem < 5:
                time.sleep(60)
            return r.json()

        return _safe_retry(_get, retries=self.retries, backoff=self.backoff)

    def fetch_scores(self, since: dt.date) -> pd.DataFrame:
        """Fetch posts from API and aggregate daily sentiment scores."""
        rows = []
        page: int | str = 1
        while True:
            data = self._fetch_posts(page)
            posts = data.get("data") or data.get("results", [])
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
            next_url = data.get("next_url") or data.get("next")
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

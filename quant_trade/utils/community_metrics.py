from __future__ import annotations

import requests
from typing import List

CM_CATALOG_URL = "https://community-api.coinmetrics.io/v4/catalog/metrics"


def community_metrics() -> List[str]:
    """Return metrics available to the CoinMetrics Community API."""
    resp = requests.get(CM_CATALOG_URL, timeout=10)
    data = resp.json()
    return [item.get("metric") for item in data.get("data", [])]


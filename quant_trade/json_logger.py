import json
from pathlib import Path
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"

def log_signal(record: Mapping[str, Any]) -> None:
    """Append a trading signal record to a JSONL file.

    The file is named ``signal_YYYYMMDD.jsonl`` under the ``logs`` directory.
    ``record`` must contain ``open_time`` field which determines the filename.
    """
    ts = record.get("open_time")
    if ts is None:
        dt = pd.Timestamp.utcnow()
    else:
        dt = pd.to_datetime(ts)
    LOG_DIR.mkdir(exist_ok=True)
    fname = LOG_DIR / f"signal_{dt.strftime('%Y%m%d')}.jsonl"
    with open(fname, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=str)
        f.write("\n")

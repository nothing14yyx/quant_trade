import os
import re
import sqlite3
from pathlib import Path

# Base directory of repository
BASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA = (BASE_DIR / 'schema.sql').read_text()

# 提取 features 表建表语句
match = re.search(r"CREATE TABLE `features` \((.*?)PRIMARY KEY", SCHEMA, re.S)
CREATE_SQL = 'CREATE TABLE features (' + match.group(1) + 'PRIMARY KEY (symbol, open_time))'

DATA_FILE = BASE_DIR / 'quant_trade' / 'data' / 'merged' / 'feature_cols.txt'
FEATURE_COLS = [c.strip() for c in DATA_FILE.read_text().splitlines() if c.strip()]
META_COLS = {'symbol', 'open_time', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base', 'taker_buy_quote'}

def test_feature_columns_match():
    conn = sqlite3.connect(':memory:')
    conn.execute(CREATE_SQL)
    cols = [r[1] for r in conn.execute('PRAGMA table_info(features)')]
    cols = [c for c in cols if c not in META_COLS]
    overlap = set(FEATURE_COLS) & set(cols)
    assert len(overlap) / len(cols) > 0.8

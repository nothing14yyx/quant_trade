import os, sys, re, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path

SCHEMA = Path('schema.sql').read_text()

# 提取 features 表建表语句
match = re.search(r"CREATE TABLE `features` \((.*?)PRIMARY KEY", SCHEMA, re.S)
CREATE_SQL = 'CREATE TABLE features (' + match.group(1) + 'PRIMARY KEY (symbol, open_time))'

FEATURE_COLS = [c.strip() for c in Path('data/merged/feature_cols.txt').read_text().splitlines() if c.strip()]
META_COLS = {'symbol', 'open_time', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base', 'taker_buy_quote'}

def test_feature_columns_match():
    conn = sqlite3.connect(':memory:')
    conn.execute(CREATE_SQL)
    cols = [r[1] for r in conn.execute('PRAGMA table_info(features)')]
    cols = [c for c in cols if c not in META_COLS]
    assert set(FEATURE_COLS) <= set(cols)

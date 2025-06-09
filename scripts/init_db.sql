-- Quant Trade 数据库初始化脚本
-- v2.4 移除 depth_snapshot 表

CREATE TABLE IF NOT EXISTS sentiment (
    timestamp BIGINT PRIMARY KEY,
    value VARCHAR(10),
    value_classification VARCHAR(20),
    fg_index DOUBLE
);

CREATE TABLE IF NOT EXISTS funding_rate (
    symbol VARCHAR(20),
    fundingTime DATETIME,
    fundingRate DOUBLE,
    PRIMARY KEY(symbol, fundingTime)
);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol VARCHAR(20),
    timestamp DATETIME,
    open_interest DOUBLE,
    PRIMARY KEY(symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS cg_market_data (
    symbol VARCHAR(20),
    timestamp DATETIME,
    price DOUBLE,
    market_cap DOUBLE,
    total_volume DOUBLE,
    PRIMARY KEY(symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS cg_global_metrics (
    timestamp DATETIME PRIMARY KEY,
    total_market_cap DOUBLE,
    total_volume DOUBLE,
    btc_dominance DOUBLE,
    eth_dominance DOUBLE
);

CREATE TABLE IF NOT EXISTS klines (
    symbol VARCHAR(20),
    `interval` VARCHAR(10),
    open_time DATETIME,
    close_time DATETIME,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    quote_asset_volume DOUBLE,
    num_trades INT,
    taker_buy_base DOUBLE,
    taker_buy_quote DOUBLE,
    fg_index DOUBLE,
    funding_rate DOUBLE,
    cg_price DOUBLE,
    cg_market_cap DOUBLE,
    cg_total_volume DOUBLE,
    PRIMARY KEY(symbol, `interval`, open_time)
);

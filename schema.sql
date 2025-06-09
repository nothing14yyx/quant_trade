CREATE TABLE cg_market_data (
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    price DOUBLE,
    market_cap DOUBLE,
    total_volume DOUBLE,
    PRIMARY KEY (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE cg_global_metrics (
    timestamp DATETIME(3) NOT NULL PRIMARY KEY,
    total_market_cap DOUBLE,
    total_volume DOUBLE,
    btc_dominance DOUBLE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

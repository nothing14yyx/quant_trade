-- ───────── klines 完整建表语句 ─────────
CREATE TABLE klines (
    -- 基础信息
    symbol             VARCHAR(20)     NOT NULL,
    `interval`         VARCHAR(10)     NOT NULL,
    open_time          DATETIME(3)     NOT NULL,
    close_time         DATETIME(3)     NOT NULL,

    -- 原生 K 线
    `open`             DOUBLE,
    high               DOUBLE,
    low                DOUBLE,
    `close`            DOUBLE,
    volume             DOUBLE,
    quote_asset_volume DOUBLE,
    num_trades         INT,
    taker_buy_base     DOUBLE,
    taker_buy_quote    DOUBLE,

    -- 情绪 & 资金费率
    fg_index           DOUBLE  NULL,
    funding_rate       DOUBLE  NULL,
    cg_price           DOUBLE,
    cg_market_cap      DOUBLE,
    cg_total_volume    DOUBLE,

    -- 索引
    PRIMARY KEY (symbol, `interval`, open_time),
    INDEX idx_symbol_interval_time (symbol, `interval`, open_time),
    INDEX idx_open_time (open_time)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

CREATE TABLE sentiment (
    timestamp              BIGINT UNSIGNED NOT NULL PRIMARY KEY,  -- UTC 秒
    value                  VARCHAR(10),
    value_classification   VARCHAR(32),
    fg_index               DOUBLE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE funding_rate (
    symbol          VARCHAR(32) NOT NULL,
    fundingTime     DATETIME    NOT NULL,
    fundingRate     DOUBLE      NOT NULL,
    PRIMARY KEY (symbol, fundingTime)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
-- 持仓量表：open_interest
CREATE TABLE open_interest (
    symbol        VARCHAR(20)  NOT NULL,
    timestamp     DATETIME(3)  NOT NULL,
    open_interest DOUBLE,
    PRIMARY KEY (symbol, timestamp),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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

CREATE TABLE `features` (
    -- 基础 K 线字段
    `symbol`              VARCHAR(24) NOT NULL,
    `open_time`           DATETIME    NOT NULL,
    `close_time`          DATETIME,
    `quote_asset_volume`  DOUBLE,
    `num_trades`          INT,
    `taker_buy_base`      DOUBLE,
    `taker_buy_quote`     DOUBLE,
    `interval`            VARCHAR(8) DEFAULT '1h',
    `open`                DOUBLE,
    `high`                DOUBLE,
    `low`                 DOUBLE,
    `close`               DOUBLE,
    `volume`              DOUBLE,
    `fg_index`            DOUBLE,
    `funding_rate`        DOUBLE,

    -- 1h 周期特征
    `ema_diff_1h`             DOUBLE,
    `sma_10_1h`               DOUBLE,
    `pct_chg1_1h`             DOUBLE,
    `pct_chg3_1h`             DOUBLE,
    `pct_chg6_1h`             DOUBLE,
    `rsi_1h`                  DOUBLE,
    `rsi_slope_1h`            DOUBLE,
    `atr_pct_1h`              DOUBLE,
    `atr_chg_1h`              DOUBLE,
    `adx_1h`                  DOUBLE,
    `adx_delta_1h`            DOUBLE,
    `cci_1h`                  DOUBLE,
    `cci_delta_1h`            DOUBLE,
    `mfi_1h`                  DOUBLE,
    `bb_width_1h`             DOUBLE,
    `bb_width_chg_1h`         DOUBLE,
    `boll_perc_1h`            DOUBLE,
    `kc_perc_1h`              DOUBLE,
    `donchian_perc_1h`        DOUBLE,
    `donchian_delta_1h`       DOUBLE,
    `vol_roc_1h`              DOUBLE,
    `vol_ma_ratio_1h`         DOUBLE,
    `bull_streak_1h`          DOUBLE,
    `bear_streak_1h`          DOUBLE,
    `rsi_mul_vol_ma_ratio_1h` DOUBLE,
    `willr_1h`                DOUBLE,
    `macd_1h`                 DOUBLE,
    `macd_signal_1h`          DOUBLE,
    `macd_hist_1h`            DOUBLE,
    `obv_1h`                  DOUBLE,
    `obv_delta_1h`            DOUBLE,
    `supertrend_dir_1h`       DOUBLE,

    -- 4h 周期特征
    `ema_diff_4h`             DOUBLE,
    `sma_10_4h`               DOUBLE,
    `pct_chg1_4h`             DOUBLE,
    `pct_chg3_4h`             DOUBLE,
    `pct_chg6_4h`             DOUBLE,
    `rsi_4h`                  DOUBLE,
    `rsi_slope_4h`            DOUBLE,
    `atr_pct_4h`              DOUBLE,
    `atr_chg_4h`              DOUBLE,
    `adx_4h`                  DOUBLE,
    `adx_delta_4h`            DOUBLE,
    `cci_4h`                  DOUBLE,
    `cci_delta_4h`            DOUBLE,
    `mfi_4h`                  DOUBLE,
    `bb_width_4h`             DOUBLE,
    `bb_width_chg_4h`         DOUBLE,
    `boll_perc_4h`            DOUBLE,
    `kc_perc_4h`              DOUBLE,
    `donchian_perc_4h`        DOUBLE,
    `donchian_delta_4h`       DOUBLE,
    `vol_roc_4h`              DOUBLE,
    `vol_ma_ratio_4h`         DOUBLE,
    `bull_streak_4h`          DOUBLE,
    `bear_streak_4h`          DOUBLE,
    `rsi_mul_vol_ma_ratio_4h` DOUBLE,
    `willr_4h`                DOUBLE,
    `macd_4h`                 DOUBLE,
    `macd_signal_4h`          DOUBLE,
    `macd_hist_4h`            DOUBLE,
    `obv_4h`                  DOUBLE,
    `obv_delta_4h`            DOUBLE,
    `supertrend_dir_4h`       DOUBLE,

    -- 1d 周期特征
    `ema_diff_d1`             DOUBLE,
    `sma_10_d1`               DOUBLE,
    `pct_chg1_d1`             DOUBLE,
    `pct_chg3_d1`             DOUBLE,
    `pct_chg6_d1`             DOUBLE,
    `rsi_d1`                  DOUBLE,
    `rsi_slope_d1`            DOUBLE,
    `atr_pct_d1`              DOUBLE,
    `atr_chg_d1`              DOUBLE,
    `adx_d1`                  DOUBLE,
    `adx_delta_d1`            DOUBLE,
    `cci_d1`                  DOUBLE,
    `cci_delta_d1`            DOUBLE,
    `mfi_d1`                  DOUBLE,
    `bb_width_d1`             DOUBLE,
    `bb_width_chg_d1`         DOUBLE,
    `boll_perc_d1`            DOUBLE,
    `kc_perc_d1`              DOUBLE,
    `donchian_perc_d1`        DOUBLE,
    `donchian_delta_d1`       DOUBLE,
    `vol_roc_d1`              DOUBLE,
    `vol_ma_ratio_d1`         DOUBLE,
    `bull_streak_d1`          DOUBLE,
    `bear_streak_d1`          DOUBLE,
    `rsi_mul_vol_ma_ratio_d1` DOUBLE,
    `willr_d1`                DOUBLE,
    `macd_d1`                 DOUBLE,
    `macd_signal_d1`          DOUBLE,
    `macd_hist_d1`            DOUBLE,
    `obv_d1`                  DOUBLE,
    `obv_delta_d1`            DOUBLE,
    `supertrend_dir_d1`       DOUBLE,

    -- 跨周期与时间特征
    `close_spread_1h_4h`      DOUBLE,
    `close_spread_1h_d1`      DOUBLE,
    `ma_ratio_1h_4h`          DOUBLE,
    `ma_ratio_1h_d1`          DOUBLE,
    `atr_pct_ratio_1h_4h`     DOUBLE,
    `bb_width_ratio_1h_4h`    DOUBLE,
    `hour_of_day`             DOUBLE,
    `day_of_week`             DOUBLE,

    -- 标签
    `target_up`               INT,
    `target_down`             INT,

    PRIMARY KEY (`symbol`, `open_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `live_full_data` (
    `symbol`      VARCHAR(24) NOT NULL,
    `time`        DATETIME NOT NULL,
    `price`       DOUBLE,
    `signal`      TINYINT,
    `score`       DOUBLE,
    `pos`         DOUBLE,
    `take_profit` DOUBLE NULL,
    `stop_loss`   DOUBLE NULL,
    `indicators`  JSON,      -- 也可使用 TEXT 类型
    PRIMARY KEY (`symbol`, `time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `live_top10_signals` (
    `symbol` VARCHAR(24) NOT NULL,
    `time` DATETIME NOT NULL,
    `price` DOUBLE,
    `signal` TINYINT,
    `score` DOUBLE,
    `pos` DOUBLE,
    `take_profit` DOUBLE NULL,
    `stop_loss` DOUBLE NULL,
    `indicators`  JSON,      -- 也可使用 TEXT 类型
    PRIMARY KEY (`symbol`, `time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

TRUNCATE TABLE klines;
TRUNCATE TABLE sentiment;
TRUNCATE TABLE funding_rate;
TRUNCATE TABLE open_interest;
TRUNCATE TABLE cg_market_data;
TRUNCATE TABLE cg_global_metrics;
TRUNCATE TABLE features;
TRUNCATE TABLE live_full_data;
TRUNCATE TABLE live_top10_signals;

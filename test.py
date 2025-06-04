import os
import yaml
import time
import logging
import requests
import pandas as pd
from binance.client import Client
from sqlalchemy import create_engine
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.ratelimiter import RateLimiter


class DataLoader:

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        bin_cfg = cfg.get('binance', {})
        dl_cfg = cfg.get('data_loader', {})
        self.api_key    = os.getenv('BINANCE_API_KEY') or bin_cfg.get('api_key', '')
        self.api_secret = os.getenv('BINANCE_API_SECRET') or bin_cfg.get('api_secret', '')
        self.proxies    = bin_cfg.get('proxy', {})
        self.topn            = dl_cfg.get('topn', 20)
        self.interval        = dl_cfg.get('interval', '4h')
        self.aux_intervals   = dl_cfg.get('aux_interval', ['1h'])
        if isinstance(self.aux_intervals, str):
            self.aux_intervals = [self.aux_intervals]
        self.start           = dl_cfg.get('start', '2021-01-01')
        self.end             = dl_cfg.get('end', None)
        self.retries         = dl_cfg.get('retries', 3)
        self.backoff         = dl_cfg.get('backoff', 1)
        self.excluded_list   = dl_cfg.get('excluded_list', [])
        self.client = Client(self.api_key, self.api_secret)
        self.klines_rate_limiter = RateLimiter(max_calls=5, period=1.0)
        if self.proxies:
            self.client.session.proxies.update(self.proxies)
        mysql_cfg = cfg.get('mysql', {})
        self.mysql_engine = create_engine(
            f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}@{mysql_cfg['host']}:{mysql_cfg.get('port', 3306)}/{mysql_cfg['database']}?charset={mysql_cfg.get('charset', 'utf8mb4')}"
        )

    def _retry(self, fn, *args, **kwargs):
        for i in range(self.retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if 'Invalid symbol' in msg or 'code=-1121' in msg:
                    sym = kwargs.get("symbol", args[0] if args else "unknown")
                    # print(f"[DataLoader] {sym} 是无效币种，自动跳过！")
                    return None
                logging.warning(f"Retry {i + 1}/{self.retries} for {fn.__name__}: {e}")
                time.sleep(self.backoff * (2 ** i))
        raise RuntimeError(f"Failed after {self.retries} retries for {fn.__name__}")

    def fetch_futures_symbols(self):
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        try:
            res = requests.get(url, timeout=10).json()
            symbols = [s['symbol'] for s in res['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
            return symbols
        except Exception as e:
            print(f"拉取合约币种失败: {e}")
            return []

    def get_top_symbols(self, n=None):
        all_symbols = self.fetch_futures_symbols()
        tradable = [s for s in all_symbols if s.endswith("USDT") and s not in self.excluded_list]
        return tradable[:n or self.topn]

    def fetch_sentiment(self):
        url = 'https://api.alternative.me/fng/?limit=0&format=json'
        try:
            resp = self._retry(requests.get, url, timeout=10)
            data = resp.json().get('data', [])
            if not data:
                print("[fetch_sentiment] API返回空数据")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['fg_index'] = df['value'].astype(float) / 100.0
            df['timestamp'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp')
            return df
        except Exception as e:
            print(f"[fetch_sentiment] 拉取异常: {e}")
            return pd.DataFrame()

    def save_sentiment_to_db(self, df):
        # 确保 timestamp 为 int（秒级时间戳）
        df['timestamp'] = df['timestamp'].astype(int)
        exist = pd.read_sql("SELECT timestamp FROM sentiment", self.mysql_engine)
        exist['timestamp'] = exist['timestamp'].astype(int)
        new_df = df[~df['timestamp'].isin(exist['timestamp'])]
        allowed_cols = ['value', 'value_classification', 'timestamp', 'fg_index']
        new_df = new_df[allowed_cols]
        if not new_df.empty:
            new_df.to_sql('sentiment', self.mysql_engine, if_exists='append', index=False)
            print(f"写入 {len(new_df)} 条情绪数据")
        else:
            print("[INFO] 今日情绪数据已存在，无需重复写入")

    def fetch_funding_rate(self, symbol, start=None, end=None, save_to_db=True):
        """
        拉取币安永续合约资金费率历史
        """
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        start = pd.to_datetime(start or self.start)
        end = pd.to_datetime(end or self.end) if self.end or end else pd.Timestamp.now()
        startTime = int(start.timestamp() * 1000)
        endTime = int(end.timestamp() * 1000)
        all_rates = []
        while startTime < endTime:
            params = {
                'symbol': symbol,
                'startTime': startTime,
                'endTime': endTime,
                'limit': 1000
            }
            try:
                resp = self._retry(requests.get, url, params=params, timeout=10)
                data = resp.json()
                if not data:
                    break
                all_rates.extend(data)
                last_time = int(data[-1]['fundingTime'])
                if last_time == startTime:
                    break
                startTime = last_time + 1
                time.sleep(0.2)
            except Exception as e:
                print(f"拉取资金费率异常: {e}")
                break
        # 整理DataFrame
        if not all_rates:
            print(f"{symbol} 没有资金费率数据")
            return pd.DataFrame()
        df = pd.DataFrame(all_rates)
        df['symbol'] = symbol
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df = df[['symbol', 'fundingTime', 'fundingRate']]
        if save_to_db:
            df.to_sql('funding_rate', self.mysql_engine, if_exists='append', index=False)
            print(f"[MySQL] 已插入 {symbol} funding_rate {len(df)} 条")
        return df

    def fetch_klines(self, symbol, interval, start=None, end=None, limit=1000):
        self.klines_rate_limiter.acquire()
        start = pd.to_datetime(start or self.start)
        end = pd.to_datetime(end or self.end) if self.end or end else pd.Timestamp.now()
        klines_all = []
        last_time = int(start.timestamp() * 1000)
        end_time = int(end.timestamp() * 1000)
        while last_time < end_time:
            klines = self._retry(
                self.client.get_klines,
                symbol=symbol,
                interval=interval,
                startTime=last_time,
                endTime=end_time,
                limit=limit
            )
            if not klines:
                break
            klines_all.extend(klines)
            last_time_new = klines[-1][0]
            if last_time_new == last_time:
                break
            last_time = last_time_new + 1
            time.sleep(0.2)
        return klines_all

    def load_klines(self, symbol: str, interval: str = None) -> pd.DataFrame:
        interval = interval or self.interval
        klines = self.fetch_klines(symbol, interval)
        if klines is None or len(klines) == 0:
            print(f"未获取到 {symbol}-{interval} K线数据")
            return pd.DataFrame()

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(klines, columns=cols)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize(None)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize(None)
        float_cols = [c for c in cols if c not in ('open_time', 'close_time', 'num_trades', 'ignore')]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['num_trades'] = pd.to_numeric(df['num_trades'], errors='coerce')

        # 合并情绪
        sentiment = self.fetch_sentiment()
        if not sentiment.empty and 'fg_index' in sentiment.columns and 'timestamp' in sentiment.columns:
            sentiment['timestamp'] = pd.to_datetime(sentiment['timestamp'], unit='s')
            df = df.sort_values('open_time')
            sentiment = sentiment.sort_values('timestamp')
            df = pd.merge_asof(
                df,
                sentiment[['timestamp', 'fg_index']],
                left_on='open_time',
                right_on='timestamp',
                direction='backward'
            ).drop(columns=['timestamp'])
        else:
            df['fg_index'] = None

        # ========== 合并资金费率 funding_rate ==========
        try:
            funding = pd.read_sql(
                f"SELECT fundingTime, fundingRate FROM funding_rate WHERE symbol='{symbol}' ORDER BY fundingTime",
                self.mysql_engine, parse_dates=['fundingTime']
            )
            df = df.sort_values('open_time')
            funding = funding.sort_values('fundingTime')
            df = pd.merge_asof(
                df,
                funding,
                left_on='open_time',
                right_on='fundingTime',
                direction='backward'
            ).rename(columns={'fundingRate': 'funding_rate'})
            df = df.drop(columns=['fundingTime'])
        except Exception as e:
            df['funding_rate'] = None
        return df

    def load_klines_from_db(self, symbol: str, interval: str) -> pd.DataFrame:
        sql = f"SELECT * FROM klines WHERE symbol='{symbol}' AND `interval`='{interval}' ORDER BY open_time"
        df = pd.read_sql(sql, self.mysql_engine, parse_dates=['open_time', 'close_time'])
        return df

    def save_klines_to_db(self, df, symbol, interval):
        df = df.copy()
        df['symbol'] = symbol
        df['interval'] = interval
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])
        try:
            df.to_sql('klines', self.mysql_engine, if_exists='append', index=False)
            print(f"[MySQL] 已插入 {symbol} {interval} k线 {len(df)} 条")
        except Exception as e:
            print(f"[MySQL] 写入k线出错: {e}")

    def merge_sentiment_to_klines(self):
        import pandas as pd
        from sqlalchemy import text

        # 1. 读取klines和sentiment
        df_klines = pd.read_sql(
            "SELECT symbol, `interval`, open_time FROM klines ORDER BY symbol, `interval`, open_time",
            self.mysql_engine)
        df_sentiment = pd.read_sql("SELECT timestamp, fg_index FROM sentiment ORDER BY timestamp", self.mysql_engine)
        df_sentiment['open_time'] = pd.to_datetime(df_sentiment['timestamp'], unit='s')
        df_klines['open_time'] = pd.to_datetime(df_klines['open_time'])

        # 2. 按symbol和interval分组分别merge_asof
        dfs = []
        for (sym, interv), group_kl in df_klines.groupby(['symbol', 'interval']):
            group_sent = df_sentiment.copy()  # 情绪不分币种周期，全局用
            group_kl = group_kl.sort_values('open_time')
            group_sent = group_sent.sort_values('open_time')
            merged = pd.merge_asof(
                group_kl,
                group_sent[['open_time', 'fg_index']],
                on='open_time',
                direction='backward'
            )
            dfs.append(merged)
        df_klines = pd.concat(dfs, ignore_index=True)

        # 3. 更新回数据库
        updated = 0
        with self.mysql_engine.begin() as conn:
            for row in df_klines.itertuples():
                if pd.notna(row.fg_index):
                    conn.execute(
                        text(
                            "UPDATE klines SET fg_index=:fg_index WHERE symbol=:symbol AND `interval`=:interval AND open_time=:open_time"
                        ),
                        {
                            "fg_index": float(row.fg_index),
                            "symbol": row.symbol,
                            "interval": row.interval,
                            "open_time": row.open_time
                        }
                    )
                    updated += 1
        print(f"[merge_sentiment_to_klines] 已成功合并写回 {updated} 条fg_index")

    def incremental_update_klines(self, symbol, interval):
        sql = f"SELECT open_time FROM klines WHERE symbol='{symbol}' AND `interval`='{interval}' ORDER BY open_time DESC LIMIT 1"
        last_df = pd.read_sql(sql, self.mysql_engine, parse_dates=['open_time'])
        last_time = last_df['open_time'].iloc[0] if not last_df.empty else None

        new_klines = self.fetch_klines(symbol, interval, start=last_time)
        if not new_klines:
            # print(f"[{symbol}-{interval}] 无新K线")
            return

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(new_klines, columns=cols)
        df['symbol'] = symbol
        df['interval'] = interval
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        if last_time is not None:
            df = df[df['open_time'] > last_time]
        if df.empty:
            # print(f"[{symbol}-{interval}] 没有需要新增的K线")
            return

        # 合并情绪
        sentiment = pd.read_sql("SELECT timestamp, fg_index FROM sentiment ORDER BY timestamp", self.mysql_engine, parse_dates=['timestamp'])
        df = df.sort_values('open_time')
        sentiment = sentiment.sort_values('timestamp')
        df = pd.merge_asof(
            df,
            sentiment,
            left_on='open_time',
            right_on='timestamp',
            direction='backward'
        )
        df = df.drop(columns=['timestamp'])
        try:
            funding = pd.read_sql(
                f"SELECT fundingTime, fundingRate FROM funding_rate WHERE symbol='{symbol}' ORDER BY fundingTime",
                self.mysql_engine, parse_dates=['fundingTime']
            )
            df = df.sort_values('open_time')
            funding = funding.sort_values('fundingTime')
            df = pd.merge_asof(
                df,
                funding,
                left_on='open_time',
                right_on='fundingTime',
                direction='backward'
            ).rename(columns={'fundingRate': 'funding_rate'})
            df = df.drop(columns=['fundingTime'])
        except Exception as e:
            df['funding_rate'] = None

        df.to_sql('klines', self.mysql_engine, if_exists='append', index=False)
        # print(f"[{symbol}-{interval}] 增量同步 {len(df)} 条新K线（含情绪）")



# ---------- 程序入口 ----------

def sync_all_symbols_threaded(loader, symbols, intervals, max_workers=8):
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for symbol in symbols:
            for interval in intervals:
                tasks.append(executor.submit(loader.incremental_update_klines, symbol, interval))
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"[线程同步异常] {e}")

def main():
    # 1) 读取配置并实例化 DataLoader
    loader = DataLoader(config_path="utils/config.yaml")

    # 2) 拉取并保存情绪数据
    sentiment_df = loader.fetch_sentiment()
    if not sentiment_df.empty:
        loader.save_sentiment_to_db(sentiment_df)

    # 3) 获取 top N 币种
    symbols = loader.get_top_symbols()
    print(f"[main] 处理币种列表：{symbols}")

    # 4) 拉取并保存资金费率数据（建议在拉K线之前完成）
    for symbol in symbols:
        print(f"[main] 拉取 {symbol} funding rate ...")
        loader.fetch_funding_rate(symbol, start="2020-01-01")

    # 5) 多线程同步所有周期所有币种的K线
    periods = [loader.interval] + loader.aux_intervals  # 例：['4h','1h','1d']
    print("[main] 多线程同步所有币种K线...")
    sync_all_symbols_threaded(loader, symbols, periods, max_workers=8)

    print("[main] 全部执行完毕！")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()






import time
from datetime import datetime, timezone, timedelta
import pytz
from pathlib import Path
import os

TZ_SH = pytz.timezone("Asia/Shanghai")
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine, text
import json
import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.helper import calc_features_raw
from feature_engineering import calc_cross_features
from utils.robust_scaler import load_scaler_params_from_json, apply_robust_z_with_params
from data_loader import DataLoader
from robust_signal_generator import RobustSignalGenerator
from utils.feature_health import apply_health_check_df, health_check

# 历史窗口长度，确保长周期指标计算不产生 NaN
HISTORY_LEN = 200


# CoinGecko /global 指标最小刷新间隔（小时），按 UTC 0 点每日更新
CG_GLOBAL_INTERVAL_HOURS = 24

# 各周期K线的最小刷新频率
SYNC_SCHEDULE = {
    "5m": timedelta(minutes=15),
    "15m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}

# ==== 每日因子 IC 更新相关设置 ====
IC_UPDATE_MARGIN_MINUTES = 10  # 离 UTC0 的分钟数阈值
IC_UPDATE_WINDOW = 1000        # 历史窗口长度
IC_UPDATE_LOG = Path("data/ic_last_update.txt")

try:
    if IC_UPDATE_LOG.exists():
        _txt = IC_UPDATE_LOG.read_text().strip()
        LAST_IC_DATE = datetime.strptime(_txt, "%Y-%m-%d").date()
    else:
        LAST_IC_DATE = None
except Exception:
    LAST_IC_DATE = None


def to_shanghai(dt):
    """Convert naive or UTC datetime to Asia/Shanghai timezone."""
    if dt is None:
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_SH)

def np_encoder(obj):
    """json.dumps helper for NumPy data types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return obj


def sanitize(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return np_encoder(obj)

CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"
# ———————— 程序开头：全局初始化 ————————

# 1. 加载配置（config.yaml）并创建数据库引擎
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{os.getenv('MYSQL_PASSWORD', mysql_cfg['password'])}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port', 3306)}/{mysql_cfg['database']}?"
    f"charset={mysql_cfg.get('charset', 'utf8mb4')}"
)

# 2. 实例化 DataLoader（用于同步行情）
loader = DataLoader(config_path=str(CONFIG_PATH))

# 3. 直接从 config.yaml 读取训练时的特征列表
# 在 single_symbol_test.py 中也是通过这种方式确保包含交叉或无后缀的列
feature_cols_1h = cfg["feature_cols"]["1h"]
feature_cols_4h = cfg["feature_cols"]["4h"]
feature_cols_d1 = cfg["feature_cols"]["1d"]

# 4. 初始化 RobustSignalGenerator（使用模型路径与旧特征列表）
model_paths = {
    "1h": {
        "up": cfg["models"]["1h"]["up"],
        "down": cfg["models"]["1h"]["down"]
    },
    "4h": {
        "up": cfg["models"]["4h"]["up"],
        "down": cfg["models"]["4h"]["down"]
    },
    "d1": {
        "up": cfg["models"]["d1"]["up"],
        "down": cfg["models"]["d1"]["down"]
    },
}
signal_generator = RobustSignalGenerator(
    model_paths,
    feature_cols_1h=feature_cols_1h,
    feature_cols_4h=feature_cols_4h,
    feature_cols_d1=feature_cols_d1,
)

# 5. 加载训练时保存的通用缩放参数（1%/99% 分位）
SCALER_PATH = Path(cfg["feature_engineering"]["scaler_path"])
if not SCALER_PATH.is_file():
    raise FileNotFoundError(f"未找到 scaler 参数文件：{SCALER_PATH}")
SCALER_PARAMS = load_scaler_params_from_json(str(SCALER_PATH))

# 6. 获取 TopN 标的列表
symbols = loader.get_top_symbols()


def sync_all_symbols_threaded(loader: DataLoader, symbols: list[str], intervals: list[str], max_workers: int = 4):
    """
    并发增量更新 K 线：对每个 symbol 和 interval 同时调用 incremental_update_klines。
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(loader.incremental_update_klines, sym, intv): (sym, intv)
            for sym in symbols for intv in intervals
        }
        for future in as_completed(futures):
            sym, intv = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[线程同步异常] {sym}-{intv}: {e}")

def update_aux_data(loader: DataLoader, symbols: list[str]) -> None:
    """同步除 K 线外的其他指标"""
    try:
        loader.update_sentiment()
    except Exception as e:
        print(f"[sentiment 更新失败] {e}")

    try:
        loader.update_cg_global_metrics(min_interval_hours=CG_GLOBAL_INTERVAL_HOURS)
    except Exception as e:
        print(f"[cg_global_metrics 更新失败] {e}")

    try:
        loader.update_cg_market_data(symbols)
    except Exception as e:
        print(f"[cg_market_data 更新失败] {e}")

    try:
        loader.update_cg_coin_categories(symbols)
    except Exception as e:
        print(f"[cg_coin_categories 更新失败] {e}")

    try:
        loader.update_cg_category_stats()
    except Exception as e:
        print(f"[cg_category_stats 更新失败] {e}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_map = {
            executor.submit(loader.update_open_interest, sym): sym for sym in symbols
        }
        for f in as_completed(fut_map):
            sym = fut_map[f]
            try:
                f.result()
            except Exception as e:
                print(f"[open_interest 更新异常] {sym}: {e}")

        fut_map = {
            executor.submit(loader.update_funding_rate, sym): sym for sym in symbols
        }
        for f in as_completed(fut_map):
            sym = fut_map[f]
            try:
                f.result()
            except Exception as e:
                print(f"[funding_rate 更新异常] {sym}: {e}")

        fut_map = {
            executor.submit(loader.update_order_book, sym): sym for sym in symbols
        }
        for f in as_completed(fut_map):
            sym = fut_map[f]
            try:
                f.result()
            except Exception as e:
                print(f"[order_book 更新异常] {sym}: {e}")

def load_recent_ic_data(symbols: list[str], window: int = IC_UPDATE_WINDOW) -> pd.DataFrame:
    """读取近期历史数据并计算 1h 特征"""
    dfs: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = pd.read_sql(
                text(
                    "SELECT open_time, open, high, low, close, volume, fg_index, funding_rate, \"cg_price\", \"cg_market_cap\", \"cg_total_volume\" "
                    "FROM klines WHERE symbol=:sym AND `interval`='1h' ORDER BY open_time DESC LIMIT :lim"
                ),
                engine,
                params={"sym": sym, "lim": window},
                parse_dates=["open_time"],
            )
        except Exception as e:
            print(f"[ic_data] 读取 {sym} 失败: {e}")
            continue
        if df.empty:
            continue
        df = df.sort_values("open_time")
        feats = calc_features_raw(df, "1h")
        feats["symbol"] = sym
        feats["open_time"] = df["open_time"].values
        feats["open"] = df["open"].values
        feats["close"] = df["close"].values
        dfs.append(feats)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def maybe_update_ic_scores(now_utc: datetime, symbols: list[str]) -> None:
    """在接近 UTC0 时读取历史数据并更新因子 IC"""
    global LAST_IC_DATE
    if now_utc.hour == 0 and now_utc.minute < IC_UPDATE_MARGIN_MINUTES:
        today = now_utc.date()
        if LAST_IC_DATE == today:
            return
        df_recent = load_recent_ic_data(symbols)
        if df_recent.empty:
            return
        try:
            signal_generator.update_ic_scores(df_recent, group_by="symbol")
            LAST_IC_DATE = today
            IC_UPDATE_LOG.write_text(today.isoformat())
            print(f"[IC] 因子 IC 已更新 {today}")
        except Exception as e:
            print(f"[IC] 更新失败: {e}")

def upsert_df(df, table_name, engine, pk_cols):
    """将 DataFrame 的数据写入 MySQL（有主键冲突则更新）。"""
    if df.empty:
        return
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.astype(object).where(pd.notnull(df), None)

    cols = df.columns.tolist()
    insert_sql = f"INSERT INTO `{table_name}` ({', '.join('`'+c+'`' for c in cols)}) VALUES ({', '.join(['%s']*len(cols))})"
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in cols if c not in pk_cols)
    full_sql = f"{insert_sql} ON DUPLICATE KEY UPDATE {update_sql}"
    data = [tuple(row) for row in df.itertuples(index=False, name=None)]
    conn = engine.raw_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(full_sql, data)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[写入 MySQL 错误] {e}")
    finally:
        cursor.close()
        conn.close()

def main_loop(interval_sec: int = 60):
    last_1h_kline_time = None
    last_sync_times = {iv: None for iv in SYNC_SCHEDULE}

    while True:
        loop_start = time.time()
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(TZ_SH)

        # 1. 动态获取数据库里拥有 >=HISTORY_LEN 条1h K线的币种
        df_symbols_ok = pd.read_sql(
            f"""
            SELECT symbol FROM klines
            WHERE `interval`='1h'
            GROUP BY symbol
            HAVING COUNT(*) >= {HISTORY_LEN}
        """,
            engine,
        )
        symbols = df_symbols_ok["symbol"].tolist()

        # 在 UTC0 附近更新因子 IC
        maybe_update_ic_scores(now_utc, symbols)

        if not symbols:
            print(f"数据库没有任何币的1h K线（或都不足{HISTORY_LEN}条），等待...")
            time.sleep(interval_sec)
            continue

        # 2. 按计划同步各周期K线
        due_intervals = []
        for iv, delta in SYNC_SCHEDULE.items():
            last = last_sync_times.get(iv)
            if last is None or now_utc - last >= delta:
                due_intervals.append(iv)
                last_sync_times[iv] = now_utc
        if due_intervals:
            sync_all_symbols_threaded(loader, symbols, due_intervals, max_workers=4)

        # 如果上一根1h K线时间已知且尚未到达下一根K线收盘时间，跳过本轮信号计算
        if last_1h_kline_time is not None:
            next_kline_time = last_1h_kline_time + timedelta(hours=1)
            if now_utc < next_kline_time:
                waiting_local = datetime.now(TZ_SH)
                print(
                    f"无新1h K线，等待... {waiting_local.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                time.sleep(interval_sec)
                continue

        # 3. 更新其他指标（FG指数、CoinGecko 等）
        update_aux_data(loader, symbols)

        placeholders = ",".join(f"'{s}'" for s in symbols)

        # 4. 查询每个币种最新的1h K线 close_time
        sql = f"""
            SELECT symbol, MAX(close_time) AS close_time
            FROM klines
            WHERE symbol IN ({placeholders}) AND `interval`='1h'
              AND close_time < '{now_utc.strftime('%Y-%m-%d %H:%M:%S')}'
            GROUP BY symbol
        """
        df_last_1h = pd.read_sql(sql, engine, parse_dates=["close_time"])
        # ensure timezone-aware datetimes
        df_last_1h["close_time"] = pd.to_datetime(df_last_1h["close_time"], utc=True)

        # 有币没有最新K线就跳过
        missing = set(symbols) - set(df_last_1h["symbol"].tolist())
        if missing:
            print("这些币没有有效1h线，将自动跳过:", missing)
            symbols = [s for s in symbols if s in df_last_1h["symbol"].tolist()]
        if not symbols:
            print("本轮全部币都不可用，等待...")
            time.sleep(interval_sec)
            continue

        min_last_time = df_last_1h["close_time"].min()
        max_last_time = df_last_1h["close_time"].max()
        if min_last_time != max_last_time:
            print(
                f"有币种1h尚未收盘, min={to_shanghai(min_last_time)}, max={to_shanghai(max_last_time)}，等待..."
            )
            time.sleep(interval_sec)
            continue

        if last_1h_kline_time == min_last_time:
            waiting_local = datetime.now(TZ_SH)
            print(
                f"无新1h K线，等待... {waiting_local.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            time.sleep(interval_sec)
            continue

        last_1h_kline_time = min_last_time
        calc_start_local = datetime.now(TZ_SH)
        print(
            f"新1h K线开始计算：{calc_start_local.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        calc_start_local = datetime.now(TZ_SH)

        # 4. 拉特征
        dfs_feat = []
        for sym in symbols:
            for iv in ("1h", "4h", "1d"):
                q = text(
                    "SELECT symbol, `interval`, open_time, close_time, open, close, high, low, volume, fg_index, funding_rate "
                    "FROM klines WHERE symbol=:sym AND `interval`=:iv ORDER BY open_time DESC LIMIT :lim"
                )
                df_tmp = pd.read_sql(q, engine, params={"sym": sym, "iv": iv, "lim": HISTORY_LEN}, parse_dates=["open_time", "close_time"])
                dfs_feat.append(df_tmp)
        df_all = pd.concat(dfs_feat, ignore_index=True) if dfs_feat else pd.DataFrame()

        # 5. 按symbol+周期分组，数据不足 HISTORY_LEN 条的自动跳过
        from typing import Dict, Optional
        feat_data: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            sym: {} for sym in symbols
        }
        for sym in symbols:
            for iv in ("1h", "4h", "1d"):
                df_si = df_all[(df_all["symbol"] == sym) & (df_all["interval"] == iv)]
                if len(df_si) < HISTORY_LEN:
                    feat_data[sym][iv] = None
                else:
                    feat_data[sym][iv] = df_si.iloc[-HISTORY_LEN:].reset_index(drop=True)

        all_full_results: list[dict] = []
        all_fused_scores: list[float] = []
        feat_dicts: Dict[str, tuple[dict, dict, dict, float, dict, dict, dict]] = {}

        def process_symbol(sym: str):
            df_1h = feat_data[sym].get("1h")
            df_4h = feat_data[sym].get("4h")
            df_d1 = feat_data[sym].get("1d")
            if df_1h is None or df_4h is None or df_d1 is None:
                return None

            raw_1h = calc_features_raw(df_1h, "1h")
            raw_4h = calc_features_raw(df_4h, "4h")
            raw_d1 = calc_features_raw(df_d1, "d1")
            if raw_1h.empty or raw_4h.empty or raw_d1.empty:
                return None

            cross_df = calc_cross_features(raw_1h, raw_4h, raw_d1)
            merged = (
                df_1h.reset_index()
                .merge(cross_df, on="open_time", how="left", suffixes=("", "_feat"))
            )
            merged["symbol"] = sym
            merged["hour_of_day"] = merged["open_time"].dt.hour.astype(float)
            merged["day_of_week"] = merged["open_time"].dt.dayofweek.astype(float)

            last_raw = merged.iloc[[-1]]
            last_scaled = apply_robust_z_with_params(last_raw.copy(), SCALER_PARAMS)
            proc = apply_health_check_df(
                last_scaled,
                abs_clip={"atr_pct_1h": (0, 0.2), "atr_pct_4h": (0, 0.2), "atr_pct_d1": (0, 0.2)},
            )

            row_scaled = proc.iloc[0]
            row_raw = last_raw.iloc[0]

            feat_1h = {c: row_scaled[c] for c in feature_cols_1h if c in row_scaled}
            feat_4h = {c: row_scaled[c] for c in feature_cols_4h if c in row_scaled}
            feat_d1 = {c: row_scaled[c] for c in feature_cols_d1 if c in row_scaled}

            price_4h = df_4h["close"].iloc[-1]
            feat_1h["close"] = row_raw["close"]
            feat_4h["price"] = price_4h
            feat_d1["close"] = row_raw["close"]

            raw_dict = row_raw.to_dict()
            ob_imb = loader.get_latest_order_book_imbalance(sym)
            raw_dict["bid_ask_imbalance"] = ob_imb

            gm = loader.get_latest_cg_global_metrics()
            oi = loader.get_latest_open_interest(sym)
            ob = loader.get_latest_order_book_imbalance(sym)

            result = signal_generator.generate_signal(
                feat_1h,
                feat_4h,
                feat_d1,
                raw_features_1h=raw_dict,
                raw_features_4h=raw_dict,
                raw_features_d1=raw_dict,
                global_metrics=gm,
                open_interest=oi,
                order_book_imbalance=ob,
            )
            fused_score = result["score"]

            return (
                sym,
                fused_score,
                feat_1h,
                feat_4h,
                feat_d1,
                price_4h,
                raw_dict,
                raw_dict,
                raw_dict,
            )

        # 6. 先计算融合分数 (使用多线程提高速度)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                (
                    sym,
                    fused_score,
                    feat_1h,
                    feat_4h,
                    feat_d1,
                    price_4h,
                    raw_feat_1h,
                    raw_feat_4h,
                    raw_feat_d1,
                ) = res
                all_fused_scores.append(fused_score)
                feat_dicts[sym] = (
                    feat_1h,
                    feat_4h,
                    feat_d1,
                    price_4h,
                    raw_feat_1h,
                    raw_feat_4h,
                    raw_feat_d1,
                )

        # 7. 计算最终信号
        for sym, (
            feat_1h,
            feat_4h,
            feat_d1,
            price_4h,
            raw_feat_1h,
            raw_feat_4h,
            raw_feat_d1,
        ) in feat_dicts.items():
            df_1h = feat_data[sym]["1h"]
            kline_close_time = df_1h["close_time"].iloc[-1] if "close_time" in df_1h.columns else None

            gm = loader.get_latest_cg_global_metrics()
            oi = loader.get_latest_open_interest(sym)
            ob = loader.get_latest_order_book_imbalance(sym)

            result = signal_generator.generate_signal(
                feat_1h,
                feat_4h,
                feat_d1,
                all_scores_list=all_fused_scores,
                raw_features_1h=raw_feat_1h,
                raw_features_4h=raw_feat_4h,
                raw_features_d1=raw_feat_d1,
                global_metrics=gm,
                open_interest=oi,
                order_book_imbalance=ob,
            )
            record = {
                "symbol": sym,
                "time": kline_close_time,
                "price": feat_1h.get("close"),
                "signal": result["signal"],
                "score": result["score"],
                "pos": result.get("position_size", 0.0),
                "take_profit": result.get("take_profit"),
                "stop_loss": result.get("stop_loss"),
                "indicators": json.dumps(
                    sanitize(
                        {
                            "feat_1h": feat_1h,
                            "feat_4h": feat_4h,
                            "feat_d1": feat_d1,
                            "raw_feat_1h": raw_feat_1h,
                            "raw_feat_4h": raw_feat_4h,
                            "raw_feat_d1": raw_feat_d1,
                            **(result.get("details") or {}),
                        }
                    ),
                    default=np_encoder,
                ),
            }
            all_full_results.append(record)

        # 8. 批量写入
        if all_full_results:
            df_full = pd.DataFrame(all_full_results)

            # —— 8.1 使用 upsert 写入 live_full_data，保留历史所有条目 ——
            upsert_df(
                df_full,
                "live_full_data",
                engine,
                pk_cols=["symbol", "time"],
            )

            df_all_results = pd.DataFrame(all_full_results)
            # ——— 确保 score 为浮点数，否则 abs() 失效 ———
            df_all_results["score"] = pd.to_numeric(df_all_results["score"], errors="coerce")
            df_all_results["time"] = pd.to_datetime(df_all_results["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            df_all_results["abs_score"] = df_all_results["score"].abs()

            df_top10 = (
                df_all_results
                .sort_values("abs_score", ascending=False)
                .head(10)
                .drop(columns=["abs_score"])
            )

            # —— 8.2 写入 live_top10_signals（遇到主键冲突则更新） ——
            df_top10["time"] = pd.to_datetime(df_top10["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            print("===== 本轮 top10（按 |score| 排序） =====")
            print(df_top10[["symbol", "time", "signal", "score", "price", "take_profit", "stop_loss"]].to_string(index=False))
            print("=====================================")

            upsert_df(
                df_top10,
                "live_top10_signals",
                engine,
                pk_cols=["symbol", "time"],
            )

        elapsed = time.time() - loop_start
        wait = max(0, interval_sec - elapsed)
        finished_local = datetime.now(TZ_SH)
        next_start = finished_local + timedelta(seconds=wait)
        print(
            f"本轮完成，耗时 {elapsed:.2f} 秒，已写入信号，时间：{finished_local.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        time.sleep(wait)

if __name__ == "__main__":
    main_loop(interval_sec=60)

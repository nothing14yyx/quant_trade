import time
import pandas as pd
import numpy as np
from robust_signal_generator import RobustSignalGenerator
from data_loader import DataLoader
import yaml
import json
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from utils.helper import calc_features_full
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 导入 Robust-z 工具
from utils.robust_scaler import (
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)

# === 初始化 ===
loader = DataLoader(config_path="utils/config.yaml")
with open('utils/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
mysql_cfg = cfg['mysql']
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}@"
    f"{mysql_cfg['host']}:{mysql_cfg.get('port', 3306)}/{mysql_cfg['database']}?"
    f"charset={mysql_cfg.get('charset', 'utf8mb4')}"
)

# —— 预加载 Robust-z 参数 JSON ——#
SCALER_PATH = Path(cfg['feature_engineering']['scaler_path'])
if not SCALER_PATH.is_file():
    raise FileNotFoundError(f"未找到 scaler 参数文件：{SCALER_PATH}")
SCALER_PARAMS = load_scaler_params_from_json(str(SCALER_PATH))


def sync_all_symbols_threaded(loader, symbols, intervals, max_workers=8):
    """并发地为多个标的、多个周期执行增量更新 K 线。"""
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


def calc_latest_features(df: pd.DataFrame, period: str) -> dict | None:
    """
    提取 DataFrame df 中对应周期的最新特征，并做 Robust-z 标准化后返回字典。
    如果数据不足 60 行、或最后一行有 NaN，就返回 None。
    """
    if len(df) < 60:
        return None
    tail_df = df.tail(60)
    feats = calc_features_full(tail_df, period=period)
    if feats is None or feats.empty or feats.iloc[-1].isnull().any():
        return None

    # —— 将最后一行原始特征转成单行 DataFrame，再做 Robust-z ——#
    raw_series = feats.iloc[-1]
    single_df = raw_series.to_frame().T  # 1×N DataFrame
    scaled_df = apply_robust_z_with_params(single_df, SCALER_PARAMS)
    return scaled_df.iloc[0].to_dict()


def main_loop(interval_sec: int = 60):
    # 1. 获取所有同时存在 1h/4h/1d 数据的标的列表
    symbols = loader.get_top_symbols()
    last_sentiment_date, last_kline_time = None, {}

    # 2. 读取 feature_cols.txt，初始化 RobustSignalGenerator
    with open("data/merged/feature_cols.txt", "r", encoding="utf-8") as f:
        all_cols = [line.strip() for line in f if line.strip()]
    feature_cols_1h = [c for c in all_cols if c.endswith("_1h")]
    feature_cols_4h = [c for c in all_cols if c.endswith("_4h")]
    feature_cols_d1 = [c for c in all_cols if c.endswith("_d1")]

    model_paths = {
        "1h": {"up": "models/model_1h_up.pkl", "down": "models/model_1h_down.pkl"},
        "4h": {"up": "models/model_4h_up.pkl", "down": "models/model_4h_down.pkl"},
        "d1": {"up": "models/model_d1_up.pkl", "down": "models/model_d1_down.pkl"},
    }
    signal_generator = RobustSignalGenerator(
        model_paths, feature_cols_1h, feature_cols_4h, feature_cols_d1
    )

    # 3. SQL 模板：从 klines 表里拉数据
    sql_template = text(
        "SELECT * FROM klines WHERE symbol=:symbol AND `interval`=:interval ORDER BY open_time"
    )

    while True:
        loop_start = time.time()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 【关键：每轮信号前，先同步K线到最新！】
        sync_all_symbols_threaded(loader, symbols, ['1h', '4h', '1d'], max_workers=8)

        new_kline_time, has_new_kline = {}, False
        all_results = []

        # 4. 遍历所有标的
        for sym in symbols:
            # 4.1 拉取 1h/4h/1d K 线
            df_1h = pd.read_sql(
                sql_template,
                engine,
                params={"symbol": sym, "interval": "1h"},
                parse_dates=["open_time", "close_time"],
            )
            df_4h = pd.read_sql(
                sql_template,
                engine,
                params={"symbol": sym, "interval": "4h"},
                parse_dates=["open_time", "close_time"],
            )
            df_1d = pd.read_sql(
                sql_template,
                engine,
                params={"symbol": sym, "interval": "1d"},
                parse_dates=["open_time", "close_time"],
            )
            if df_1h.empty or df_4h.empty or df_1d.empty:
                continue

            # ====== 新增：只保留最近60根K线 ======
            df_1h = df_1h.tail(60)
            df_4h = df_4h.tail(60)
            df_1d = df_1d.tail(60)

            # 4.2 获取最新 time & price
            last_time = df_1h["open_time"].iloc[-1]
            last_price = df_1h["close"].iloc[-1]

            # 4.3 判断是否有新 K 线
            prev_time = last_kline_time.get(sym)
            if prev_time is None or last_time > prev_time:
                has_new_kline = True
            new_kline_time[sym] = last_time

            # 4.4 计算整张特征表
            feats_1h_df = calc_features_full(df_1h, "1h")
            feats_4h_df = calc_features_full(df_4h, "4h")
            feats_d1_df = calc_features_full(df_1d, "d1")
            if feats_1h_df.empty or feats_4h_df.empty or feats_d1_df.empty:
                continue

            # 4.5 取最后一行特征字典
            feat_1h = feats_1h_df.iloc[-1].to_dict()
            feat_4h = feats_4h_df.iloc[-1].to_dict()
            feat_d1 = feats_d1_df.iloc[-1].to_dict()

            feat_4h['close'] = df_4h['close'].iloc[-1]

            # 4.6 生成信号
            result = signal_generator.generate_signal(feat_1h, feat_4h, feat_d1)

            # 4.7 补充 symbol、time、price、pos
            result["symbol"] = sym
            result["time"] = last_time
            result["price"] = last_price
            result["pos"] = result.get("position_size", 0.0)
            all_results.append(result)

        # 5. 更新 last_kline_time
        last_kline_time.update(new_kline_time)

        # 6. 写入 live_full_data（只保留主字段）
        if all_results:
            df_full = pd.DataFrame(all_results)
            save_cols = [
                "symbol", "time", "price",
                "signal", "score", "pos", "take_profit", "stop_loss"
            ]
            save_cols = [col for col in save_cols if col in df_full.columns]
            df_full = df_full[save_cols]
            upsert_df(df_full, "live_full_data", engine, pk_cols=["symbol", "time"])

        # 7. 写入并打印 Top10 强信号（按绝对分数排序）
        if has_new_kline:
            if all_results:
                df_all = pd.DataFrame(all_results)
                # 按绝对分数排序
                df_top10 = df_all.reindex(df_all["score"].abs().sort_values(ascending=False).index).head(
                    10).reset_index(drop=True)
                print(f"[Top10强信号-{now}]")
                for r in df_top10.itertuples(index=False):
                    t_str = r.time.strftime("%Y-%m-%d %H:%M:%S")
                    direction = "多" if r.signal == 1 else "空"
                    stop_loss = getattr(r, "stop_loss", None)
                    take_profit = getattr(r, "take_profit", None)
                    stop_str = f"止盈={take_profit:.4f}" if take_profit is not None else "止盈=N/A"
                    loss_str = f"止损={stop_loss:.4f}" if stop_loss is not None else "止损=N/A"
                    print(f"{r.symbol} @ {t_str}: 方向={direction} "
                        f"分数={r.score:.4f} 价格={r.price:.4f} pos={r.pos:.2f} "
                        f"{stop_str} {loss_str}"
                    )

                df_top10 = df_top10[save_cols]
                upsert_df(df_top10, "live_top10_signals", engine, pk_cols=["symbol", "time"])
            else:
                print(f"[{now}] 本轮无强信号（新K线，但没有达到阈值的信号）")
        else:
            print(f"[{now}] 本轮无新K线，等待下一根K线更新...")

        print(f"本轮用时 {time.time() - loop_start:.2f} 秒")
        time.sleep(interval_sec)



if __name__ == "__main__":
    main_loop(interval_sec=60)

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.helper import calc_features_raw
from utils.robust_scaler import load_scaler_params_from_json, apply_robust_z_with_params
from data_loader import DataLoader
from robust_signal_generator import RobustSignalGenerator
from utils.feature_health import apply_health_check_df,health_check
# ———————— 程序开头：全局初始化 ————————

# 1. 加载配置（config.yaml）并创建数据库引擎
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mysql_cfg = cfg["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_cfg['user']}:{mysql_cfg['password']}"
    f"@{mysql_cfg['host']}:{mysql_cfg.get('port', 3306)}/{mysql_cfg['database']}?"
    f"charset={mysql_cfg.get('charset', 'utf8mb4')}"
)

# 2. 实例化 DataLoader（用于同步行情）
loader = DataLoader(config_path="utils/config.yaml")

# 3. 读取 feature_cols.txt，为 RobustSignalGenerator 构造时保留旧特征列表
with open("data/merged/feature_cols.txt", "r", encoding="utf-8") as f:
    all_cols = [line.strip() for line in f if line.strip()]
feature_cols_1h = [c for c in all_cols if c.endswith("_1h")]
feature_cols_4h = [c for c in all_cols if c.endswith("_4h")]
feature_cols_d1 = [c for c in all_cols if c.endswith("_d1")]

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


def sync_all_symbols_threaded(loader: DataLoader, symbols: list[str], intervals: list[str], max_workers: int = 8):
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

    while True:
        loop_start = time.time()
        now = datetime.now(timezone.utc)

        # 1. 动态获取数据库里拥有 >=60 条1h K线的币种
        df_symbols_ok = pd.read_sql("""
            SELECT symbol FROM klines
            WHERE `interval`='1h'
            GROUP BY symbol
            HAVING COUNT(*) >= 60
        """, engine)
        symbols = df_symbols_ok["symbol"].tolist()

        if not symbols:
            print("数据库没有任何币的1h K线（或都不足60条），等待...")
            time.sleep(interval_sec)
            continue

        # 2. 多线程同步K线
        sync_all_symbols_threaded(loader, symbols, ["1h", "4h", "1d"], max_workers=8)

        placeholders = ",".join(f"'{s}'" for s in symbols)

        # 3. 查询每个币种最新的1h K线 close_time
        sql = f"""
            SELECT symbol, MAX(close_time) AS close_time
            FROM klines
            WHERE symbol IN ({placeholders}) AND `interval`='1h'
              AND close_time < '{now.strftime('%Y-%m-%d %H:%M:%S')}'
            GROUP BY symbol
        """
        df_last_1h = pd.read_sql(sql, engine, parse_dates=["close_time"])

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
            print(f"有币种1h尚未收盘, min={min_last_time}, max={max_last_time}，等待...")
            time.sleep(interval_sec)
            continue

        if last_1h_kline_time == min_last_time:
            print(f"无新1h K线，等待... {min_last_time}")
            time.sleep(interval_sec)
            continue

        last_1h_kline_time = min_last_time

        # 4. 拉特征
        sql_feat = f"""
            SELECT symbol, `interval`, open_time, close_time, open, close, high, low, volume, fg_index, funding_rate
            FROM klines
            WHERE symbol IN ({placeholders})
              AND `interval` IN ('1h','4h','1d')
            ORDER BY symbol, `interval`, open_time
        """
        df_all = pd.read_sql(sql_feat, engine, parse_dates=["open_time", "close_time"])

        # 5. 按symbol+周期分组，数据不足60条的自动跳过
        from typing import Dict, Optional
        feat_data: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            sym: {} for sym in symbols
        }
        for sym in symbols:
            for iv in ("1h", "4h", "1d"):
                df_si = df_all[(df_all["symbol"] == sym) & (df_all["interval"] == iv)]
                if len(df_si) < 60:
                    feat_data[sym][iv] = None
                else:
                    feat_data[sym][iv] = df_si.iloc[-60:].reset_index(drop=True)

        all_full_results: list[dict] = []
        all_fused_scores: list[float] = []
        feat_dicts: Dict[str, tuple[dict, dict, dict, float, dict, dict, dict]] = {}

        # 6. 先计算融合分数
        for sym in symbols:
            df_1h = feat_data[sym].get("1h")
            df_4h = feat_data[sym].get("4h")
            df_d1 = feat_data[sym].get("1d")
            if df_1h is None or df_4h is None or df_d1 is None:
                continue

            raw_1h = calc_features_raw(df_1h, "1h")
            raw_4h = calc_features_raw(df_4h, "4h")
            raw_d1 = calc_features_raw(df_d1, "d1")
            if raw_1h.empty or raw_4h.empty or raw_d1.empty:
                continue

            last_raw_1h = raw_1h.iloc[[-1]]
            last_raw_4h = raw_4h.iloc[[-1]]
            last_raw_d1 = raw_d1.iloc[[-1]]

            scaled_1h = apply_robust_z_with_params(last_raw_1h, SCALER_PARAMS)
            scaled_4h = apply_robust_z_with_params(last_raw_4h, SCALER_PARAMS)
            scaled_d1 = apply_robust_z_with_params(last_raw_d1, SCALER_PARAMS)

            proc_1h = apply_health_check_df(scaled_1h, abs_clip={"atr_pct_1h": (0, 0.2)})
            proc_4h = apply_health_check_df(scaled_4h, abs_clip={"atr_pct_4h": (0, 0.2)})
            proc_d1 = apply_health_check_df(scaled_d1, abs_clip={"atr_pct_d1": (0, 0.2)})


            feat_1h = proc_1h.iloc[0].to_dict()
            feat_4h = proc_4h.iloc[0].to_dict()
            feat_d1 = proc_d1.iloc[0].to_dict()

            raw_feat_1h = last_raw_1h.iloc[0].to_dict()
            raw_feat_4h = last_raw_4h.iloc[0].to_dict()
            raw_feat_d1 = last_raw_d1.iloc[0].to_dict()

            feat_1h = health_check(feat_1h, abs_clip={"atr_pct_1h": (0, 0.2)})
            feat_4h = health_check(feat_4h, abs_clip={"atr_pct_4h": (0, 0.2)})
            feat_d1 = health_check(feat_d1, abs_clip={"atr_pct_d1": (0, 0.2)})


            price_4h = df_4h["close"].iloc[-1]
            feat_1h["close"] = df_1h["close"].iloc[-1]
            feat_4h["price"] = price_4h
            feat_d1["close"] = df_d1["close"].iloc[-1]

            result = signal_generator.generate_signal(
                feat_1h,
                feat_4h,
                feat_d1,
                raw_features_1h=raw_feat_1h,
                raw_features_4h=raw_feat_4h,
                raw_features_d1=raw_feat_d1,
            )
            fused_score = result["score"]
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

            result = signal_generator.generate_signal(
                feat_1h,
                feat_4h,
                feat_d1,
                all_scores_list=all_fused_scores,
                raw_features_1h=raw_feat_1h,
                raw_features_4h=raw_feat_4h,
                raw_features_d1=raw_feat_d1,
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
        print(f"本轮完成，已写入信号，时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")

        time.sleep(wait)

if __name__ == "__main__":
    main_loop(interval_sec=60)

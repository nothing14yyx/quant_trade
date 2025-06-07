import os
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine
from robust_signal_generator import RobustSignalGenerator
from utils.helper import calc_features_raw

# ====== 配置特征字段（从 config.yaml 读取）======
with open("utils/config.yaml", "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
FEATURE_COLS_1H = _cfg.get("feature_cols", {}).get("1h", [])
FEATURE_COLS_4H = _cfg.get("feature_cols", {}).get("4h", [])
FEATURE_COLS_D1 = _cfg.get("feature_cols", {}).get("d1", [])

# 预训练模型路径
MODEL_PATHS = {
    ('1h', 'up'):   'models/model_1h_up.pkl',
    ('1h', 'down'): 'models/model_1h_down.pkl',
    ('4h', 'up'):   'models/model_4h_up.pkl',
    ('4h', 'down'): 'models/model_4h_down.pkl',
    ('d1', 'up'):   'models/model_d1_up.pkl',
    ('d1', 'down'): 'models/model_d1_down.pkl',
}

# 将上面的 (period, direction) 键值对转换为嵌套字典
def convert_model_paths(paths: dict) -> dict:
    nested = {}
    for (period, direction), p in paths.items():
        nested.setdefault(period, {})[direction] = p
    return nested

# =========== 数据库&配置 ===========
def load_config(path='utils/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
def connect_mysql(cfg):
    mysql = cfg['mysql']
    url = (
        f"mysql+pymysql://{mysql['user']}:{mysql['password']}@{mysql['host']}:{mysql.get('port',3306)}/{mysql['database']}?charset=utf8mb4"
    )
    return create_engine(url)


def simulate_trades(df_sym: pd.DataFrame, sig_df: pd.DataFrame, *, fee_rate: float, slippage: float) -> pd.DataFrame:
    """根据信号和K线计算回测交易明细"""
    trades = []
    in_pos = False
    entry_price = entry_time = pos_size = score = direction = tp = sl = None
    for i in range(1, len(df_sym)):
        if not in_pos:
            if i-1 < len(sig_df) and sig_df.at[i-1, 'signal'] != 0:
                direction = sig_df.at[i-1, 'signal']
                entry_price = df_sym.at[i, 'open'] * (1 + slippage * direction)
                entry_time = df_sym.at[i, 'open_time']
                pos_size = sig_df.at[i-1, 'position_size']
                score = sig_df.at[i-1, 'score']
                tp = sig_df.at[i-1, 'take_profit']
                sl = sig_df.at[i-1, 'stop_loss']
                in_pos = True
            continue

        high = df_sym.at[i, 'high']
        low = df_sym.at[i, 'low']
        exit_price = None
        exit_time = None
        if direction == 1:
            if low <= sl:
                exit_price = sl
                exit_time = df_sym.at[i, 'open_time']
            elif high >= tp:
                exit_price = tp
                exit_time = df_sym.at[i, 'open_time']
        else:
            if high >= sl:
                exit_price = sl
                exit_time = df_sym.at[i, 'open_time']
            elif low <= tp:
                exit_price = tp
                exit_time = df_sym.at[i, 'open_time']

        if exit_price is None and i < len(sig_df) and sig_df.at[i, 'signal'] == -direction:
            exit_price = df_sym.at[i, 'close'] * (1 - slippage * direction)
            exit_time = df_sym.at[i, 'close_time']

        if exit_price is not None:
            pnl = (exit_price - entry_price) * direction * pos_size
            ret = pnl / entry_price - 2 * fee_rate
            holding_s = (exit_time - entry_time).total_seconds()
            trades.append({
                'symbol': df_sym.at[i, 'symbol'],
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'signal': direction,
                'score': score,
                'position_size': pos_size,
                'pnl': pnl,
                'ret': ret,
                'holding_s': holding_s
            })
            in_pos = False

    if in_pos:
        exit_price = df_sym.at[len(df_sym)-1, 'close'] * (1 - slippage * direction)
        exit_time = df_sym.at[len(df_sym)-1, 'close_time']
        pnl = (exit_price - entry_price) * direction * pos_size
        ret = pnl / entry_price - 2 * fee_rate
        holding_s = (exit_time - entry_time).total_seconds()
        trades.append({
            'symbol': df_sym.at[len(df_sym)-1, 'symbol'],
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'signal': direction,
            'score': score,
            'position_size': pos_size,
            'pnl': pnl,
            'ret': ret,
            'holding_s': holding_s
        })

    return pd.DataFrame(trades)

# =========== 融合信号回测 ===========
def run_backtest(*, recent_days: int | None = None):
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql('SELECT * FROM features', engine, parse_dates=['open_time','close_time'])
    if recent_days:
        end_time = df['open_time'].max()
        start_time = end_time - pd.Timedelta(days=recent_days)
        df = df[df['open_time'] >= start_time]

    # 按币种分组
    all_symbols = df['symbol'].unique().tolist()
    sg = RobustSignalGenerator(

        model_paths=convert_model_paths(MODEL_PATHS),

        feature_cols_1h=FEATURE_COLS_1H,
        feature_cols_4h=FEATURE_COLS_4H,
        feature_cols_d1=FEATURE_COLS_D1,
    )

    # 根据近期历史数据更新因子 IC 分数
    sg.update_ic_scores(df.tail(1000))

    results = []
    trades_all = []

    # 参数
    fee_rate = 0.0005
    slippage = 0.0003

    for symbol in all_symbols:
        df_sym = df[df['symbol'] == symbol].copy()
        df_sym = df_sym.sort_values('open_time')
        df_sym = df_sym.reset_index(drop=True)
        # 补齐多周期特征
        for col in FEATURE_COLS_1H:
            if col not in df_sym: df_sym[col] = np.nan
        for col in FEATURE_COLS_4H:
            if col not in df_sym: df_sym[col] = np.nan
        for col in FEATURE_COLS_D1:
            if col not in df_sym: df_sym[col] = np.nan

        # === 计算未归一化的原始特征，用于动态阈值和 ATR 等逻辑 ===
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'fg_index' in df_sym.columns:
            base_cols.append('fg_index')
        if 'funding_rate' in df_sym.columns:
            base_cols.append('funding_rate')

        df_raw = df_sym[['open_time'] + base_cols].copy()
        df_raw.set_index('open_time', inplace=True)

        raw_1h_df = calc_features_raw(df_raw, '1h')

        agg = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }
        if 'fg_index' in df_raw.columns:
            agg['fg_index'] = 'last'
        if 'funding_rate' in df_raw.columns:
            agg['funding_rate'] = 'last'

        df_4h = df_raw.resample('4h', label='left', closed='left').agg(agg).dropna()
        raw_4h_df = calc_features_raw(df_4h, '4h')

        df_d1 = df_raw.resample('1d', label='left', closed='left').agg(agg).dropna()
        raw_d1_df = calc_features_raw(df_d1, 'd1')

        # 回测主循环：每根K线跑融合信号
        signals = []
        for i in range(1, len(df_sym)):   # i=0没法t+1建仓
            feats_1h = {c: df_sym.at[i, c] for c in FEATURE_COLS_1H}
            feats_4h = {c: df_sym.at[i, c] for c in FEATURE_COLS_4H}
            feats_d1 = {c: df_sym.at[i, c] for c in FEATURE_COLS_D1}

            ts = df_sym.at[i, 'open_time']
            raw1h = raw_1h_df.loc[ts].to_dict() if ts in raw_1h_df.index else {}
            r4 = raw_4h_df.loc[:ts]
            raw4h = r4.iloc[-1].to_dict() if not r4.empty else {}
            r1d = raw_d1_df.loc[:ts]
            rawd1 = r1d.iloc[-1].to_dict() if not r1d.empty else {}

            result = sg.generate_signal(
                feats_1h,
                feats_4h,
                feats_d1,
                raw_features_1h=raw1h,
                raw_features_4h=raw4h,
                raw_features_d1=rawd1,
            )
            signals.append({
                'open_time': df_sym.at[i, 'open_time'],
                'signal': result['signal'],
                'score': result['score'],
                'position_size': result['position_size'],
                'take_profit': result['take_profit'],
                'stop_loss': result['stop_loss'],
                'details': result['details'],
            })
        sig_df = pd.DataFrame(signals)
        sig_df['symbol'] = symbol

        trades_df = simulate_trades(df_sym, sig_df, fee_rate=fee_rate, slippage=slippage)
        trades_all.append(trades_df)

        # 汇总绩效
        if trades_df.empty:
            summary = {
                'symbol': symbol, 'n_trades': 0, 'total_ret': 0, 'win_rate': np.nan,
                'avg_pnl': np.nan, 'avg_win': np.nan, 'avg_loss': np.nan,
                'sharpe': np.nan, 'max_dd': np.nan, 'avg_hold_s': np.nan
            }
        else:
            n = len(trades_df)
            weights = trades_df['position_size']
            cumprod = (trades_df['ret'] + 1.0).cumprod()
            total_ret = cumprod.iloc[-1] - 1.0

            win_mask = trades_df['ret'] > 0
            loss_mask = ~win_mask

            win_rate = weights[win_mask].sum() / weights.sum() if weights.sum() != 0 else np.nan
            avg_pnl = np.average(trades_df['pnl'], weights=weights)
            avg_win = np.average(trades_df.loc[win_mask, 'pnl'], weights=weights[win_mask]) if win_mask.any() else 0
            avg_loss = np.average(trades_df.loc[loss_mask, 'pnl'], weights=weights[loss_mask]) if loss_mask.any() else 0

            hwm = cumprod.cummax()
            drawdown = cumprod / hwm - 1.0
            max_dd = drawdown.min()

            weighted_ret = np.average(trades_df['ret'], weights=weights)
            ret_var = np.average((trades_df['ret'] - weighted_ret) ** 2, weights=weights)
            ret_std = np.sqrt(ret_var)
            sharpe = weighted_ret / ret_std * np.sqrt(n) if ret_std else np.nan

            avg_hold = np.average(trades_df['holding_s'], weights=weights)
            summary = {
                'symbol': symbol, 'n_trades': n, 'total_ret': total_ret, 'win_rate': win_rate,
                'avg_pnl': avg_pnl, 'avg_win': avg_win, 'avg_loss': avg_loss,
                'sharpe': sharpe, 'max_dd': max_dd, 'avg_hold_s': avg_hold
            }
        results.append(summary)

        # 保存每个币种明细
        trades_df.to_csv(f'backtest_logs/{symbol}_fusion_trades.csv', index=False)
        print(f"{symbol} 回测完成，信号数：{len(trades_df)}")

    # 汇总
    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_fusion_summary.csv', index=False)
    print("========== 回测汇总 ==========")
    print(results_df.to_string(index=False))

    # 所有明细也合并导出一份（可选）
    all_trades = pd.concat(trades_all, ignore_index=True)
    all_trades.to_csv('backtest_fusion_trades_all.csv', index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='回测融合信号策略')
    parser.add_argument('--recent-days', type=int, default=None, help='只回测最近 N 天的数据')
    args = parser.parse_args()

    os.makedirs('backtest_logs', exist_ok=True)
    run_backtest(recent_days=args.recent_days)

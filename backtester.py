import os
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine
from robust_signal_generator import RobustSignalGenerator

# ====== 配置特征字段（和实盘一致！）======
FEATURE_COLS_1H = [
    'atr_pct_1h',  # 1h 周期波动率（ATR）
    'rsi_slope_1h',  # 1h 周期 RSI 斜率（动量变化）
    'kc_perc_1h',  # 1h 周期 Keltner 通道位置（波动位置）
    'cci_delta_1h',  # 1h 周期 CCI 变化率（顺势动量）
    'vol_ma_ratio_1h',  # 1h 周期 成交量/MA（短期量能变化）
    'rsi_mul_vol_ma_ratio_1h',  # 1h 周期 RSI × (成交量/MA)（动量与量能交互）
    'vol_roc_1h',  # 1h 周期 成交量 ROC（量能动量）
    'adx_1h',  # 1h 周期 趋势强度（ADX）
]
FEATURE_COLS_4H = [
    'atr_pct_4h',  # 4h 周期波动率（ATR）
    'vol_ma_ratio_4h',  # 4h 周期 成交量/MA（中期量能变化）
    'vol_roc_4h',  # 4h 周期 成交量 ROC（中期量能动量）
    'rsi_slope_4h',  # 4h 周期 RSI 斜率（动量变化）
    'cci_delta_4h',  # 4h 周期 CCI 变化率（顺势动量）
    'rsi_mul_vol_ma_ratio_4h',  # 4h 周期 RSI × (成交量/MA)（动量与量能交互）
    'adx_4h',  # 4h 周期 趋势强度（ADX）
    'boll_perc_4h',  # 4h 周期 布林带位置（价格相对布林带）
]
FEATURE_COLS_D1 = [
    'atr_pct_d1',  # 日线波动率（ATR）
    'rsi_slope_d1',  # 日线 RSI 斜率（动量变化）
    'rsi_mul_vol_ma_ratio_d1',  # 日线 RSI × (成交量/MA)（动量与量能交互）
    'vol_ma_ratio_d1',  # 日线 成交量/MA（量能变化）
    'vol_roc_d1',  # 日线 成交量 ROC（量能动量）
    'cci_delta_d1',  # 日线 CCI 变化率（顺势动量）
    'adx_d1',  # 日线 趋势强度（ADX）
    'boll_perc_d1',  # 日线 布林带位置（价格相对布林带）
]

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

# =========== 融合信号回测 ===========
def run_backtest():
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql('SELECT * FROM features', engine, parse_dates=['open_time','close_time'])

    # 按币种分组
    all_symbols = df['symbol'].unique().tolist()
    sg = RobustSignalGenerator(
        convert_model_paths(MODEL_PATHS),
        feature_cols_1h=FEATURE_COLS_1H,
        feature_cols_4h=FEATURE_COLS_4H,
        feature_cols_d1=FEATURE_COLS_D1
    )

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

        # 回测主循环：每根K线跑融合信号
        signals = []
        for i in range(1, len(df_sym)):   # i=0没法t+1建仓
            feats_1h = {c: df_sym.at[i, c] for c in FEATURE_COLS_1H}
            feats_4h = {c: df_sym.at[i, c] for c in FEATURE_COLS_4H}
            feats_d1 = {c: df_sym.at[i, c] for c in FEATURE_COLS_D1}
            result = sg.generate_signal(feats_1h, feats_4h, feats_d1)
            signals.append({
                'open_time': df_sym.at[i, 'open_time'],
                'signal': result['signal'],
                'score': result['score'],
                'position_size': result['position_size'],
                'details': result['details'],
            })
        sig_df = pd.DataFrame(signals)
        sig_df['symbol'] = symbol

        # 只保留有信号的行，用t+1建仓/平仓
        valid_idx = sig_df[sig_df['signal'] != 0].index + 1  # t+1
        valid_idx = valid_idx[valid_idx < len(df_sym)]
        trades = []
        for idx in valid_idx:
            entry_time  = df_sym.at[idx, 'open_time']
            entry_price = df_sym.at[idx, 'open'] * (1 + slippage * np.sign(sig_df.at[idx-1, 'signal']))
            exit_time   = df_sym.at[idx, 'open_time']
            exit_price  = df_sym.at[idx, 'close'] * (1 - slippage * np.sign(sig_df.at[idx-1, 'signal']))
            direction   = sig_df.at[idx-1, 'signal']
            # 止损止盈可选（此处不做，建议后续加）
            pnl = (exit_price - entry_price) * direction
            ret = pnl / entry_price - 2 * fee_rate
            holding_s = (exit_time - entry_time).total_seconds()
            score = sig_df.at[idx-1, 'score']
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'signal': direction,
                'score': score,
                'pnl': pnl,
                'ret': ret,
                'holding_s': holding_s
            })
        trades_df = pd.DataFrame(trades)
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
            cumprod = (trades_df['ret'] + 1.0).cumprod()
            total_ret = cumprod.iloc[-1] - 1.0
            wins = trades_df[trades_df['ret'] > 0]['ret']
            losses = trades_df[trades_df['ret'] <= 0]['ret']
            win_rate = len(wins) / n
            avg_pnl = trades_df['pnl'].mean()
            avg_win = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            hwm = cumprod.cummax()
            drawdown = cumprod / hwm - 1.0
            max_dd = drawdown.min()
            ret_std = trades_df['ret'].std()
            sharpe = trades_df['ret'].mean() / ret_std * np.sqrt(n) if ret_std else np.nan
            avg_hold = trades_df['holding_s'].mean()
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
    os.makedirs('backtest_logs', exist_ok=True)
    run_backtest()

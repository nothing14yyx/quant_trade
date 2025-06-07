import os
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine
from robust_signal_generator import RobustSignalGenerator
from utils.helper import calc_features_raw

# ====== 配置特征字段（和实盘一致！）======
FEATURE_COLS_1H = [
    'atr_pct_1h',  # 1h 波动率（ATR百分比）
    'rsi_slope_1h',  # 1h RSI斜率（动量变化）
    'kc_perc_1h',  # 1h Keltner通道分位（趋势/顺势）
    'vol_ma_ratio_1h',  # 1h 成交量/均线（量能）
    'boll_perc_1h',  # 1h 布林分位（价格偏离度）
    'fg_index',  # 日度情绪（恐惧贪婪）
    'funding_rate',  # 资金费率
    'cci_delta_1h',  # 1h CCI变化（顺势波动）
]
FEATURE_COLS_4H = [
    'atr_pct_4h',  # 4h 波动率
    'rsi_slope_4h',  # 4h RSI斜率
    'kc_perc_4h',  # 4h Keltner通道分位
    'vol_ma_ratio_4h',  # 4h 成交量/均线
    'boll_perc_4h',  # 4h 布林分位
    'fg_index_d1',  # 日度情绪（恐惧贪婪）
    'funding_rate_4h',  # 4h 资金费率
    'cci_delta_4h',  # 4h CCI变化
]
FEATURE_COLS_D1 = [
    'atr_pct_d1',  # 日线波动率
    'rsi_slope_d1',  # 日线RSI斜率
    'kc_perc_d1',  # 日线Keltner通道分位
    'vol_ma_ratio_d1',  # 日线量能/均线
    'boll_perc_d1',  # 日线布林分位
    'fg_index_d1',  # 日线情绪
    'funding_rate_d1',  # 日线资金费率
    'cci_delta_d1',  # 日线CCI变化
]

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

# =========== 融合信号回测 ===========
def run_backtest():
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql('SELECT * FROM features', engine, parse_dates=['open_time','close_time'])

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

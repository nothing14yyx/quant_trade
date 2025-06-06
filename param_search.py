import os
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine
from sklearn.model_selection import ParameterGrid

from robust_signal_generator import RobustSignalGenerator
from backtester import (

    FEATURE_COLS_1H,
    FEATURE_COLS_4H,
    FEATURE_COLS_D1,
    MODEL_PATHS,
    load_config,
    connect_mysql,

def compute_ic_scores(df: pd.DataFrame, rsg: RobustSignalGenerator) -> dict:
    """利用历史特征与未来收益率计算各因子IC"""
    df = df.sort_values('open_time').reset_index(drop=True)
    # 下一根K线的区间收益率 (open->close)
    returns = df['close'].shift(-1) / df['open'].shift(-1) - 1
    scores = {k: [] for k in rsg.base_weights.keys()}
    for i in range(len(df) - 1):
        feats = {c: df.at[i, c] for c in FEATURE_COLS_1H if c in df}
        ai = rsg.get_ai_score(feats, rsg.models['1h']['up']) - rsg.get_ai_score(feats, rsg.models['1h']['down'])
        factors = rsg.get_factor_scores(feats, '1h')
        data = {'ai': ai, **factors}
        for k in scores:
            scores[k].append(data.get(k, np.nan))
    rets = returns.iloc[:-1].values
    ic_scores = {}
    for k, vals in scores.items():
        arr = np.array(vals, dtype=float)
        mask = ~np.isnan(arr) & ~np.isnan(rets)
        if mask.sum() > 1:
            ic = np.corrcoef(arr[mask], rets[mask])[0, 1]
        else:
            ic = 0.0
        ic_scores[k] = ic
    return ic_scores


def run_single_backtest(df: pd.DataFrame, base_weights: dict, history_window: int, th_params: dict):
    """在给定参数下执行一次回测并返回总体绩效"""
    sg = RobustSignalGenerator(
        model_paths=convert_model_paths(MODEL_PATHS),
        feature_cols_1h=FEATURE_COLS_1H,
        feature_cols_4h=FEATURE_COLS_4H,
        feature_cols_d1=FEATURE_COLS_D1,
        history_window=history_window,
    )
    sg.base_weights = base_weights

    if th_params:
        def dyn_th(self, atr, adx, funding=0):
            return RobustSignalGenerator.dynamic_threshold(self, atr, adx, funding, **th_params)
        sg.dynamic_threshold = dyn_th.__get__(sg, RobustSignalGenerator)

    sg.ic_scores.update(compute_ic_scores(df, sg))

    all_symbols = df['symbol'].unique().tolist()
    fee_rate = 0.0005
    slippage = 0.0003
    results = []

    for symbol in all_symbols:
        df_sym = df[df['symbol'] == symbol].copy()
        df_sym = df_sym.sort_values('open_time').reset_index(drop=True)
        for col in FEATURE_COLS_1H:
            if col not in df_sym:
                df_sym[col] = np.nan
        for col in FEATURE_COLS_4H:
            if col not in df_sym:
                df_sym[col] = np.nan
        for col in FEATURE_COLS_D1:
            if col not in df_sym:
                df_sym[col] = np.nan

        signals = []
        for i in range(1, len(df_sym)):
            f1 = {c: df_sym.at[i, c] for c in FEATURE_COLS_1H}
            f4 = {c: df_sym.at[i, c] for c in FEATURE_COLS_4H}
            fd = {c: df_sym.at[i, c] for c in FEATURE_COLS_D1}
            res = sg.generate_signal(f1, f4, fd)
            signals.append({'open_time': df_sym.at[i, 'open_time'], 'signal': res['signal'], 'score': res['score']})
        sig_df = pd.DataFrame(signals)
        valid_idx = sig_df[sig_df['signal'] != 0].index + 1
        valid_idx = valid_idx[valid_idx < len(df_sym)]
        trades = []
        for idx in valid_idx:
            entry_price = df_sym.at[idx, 'open'] * (1 + slippage * np.sign(sig_df.at[idx-1, 'signal']))
            exit_price = df_sym.at[idx, 'close'] * (1 - slippage * np.sign(sig_df.at[idx-1, 'signal']))
            direction = sig_df.at[idx-1, 'signal']
            pnl = (exit_price - entry_price) * direction
            ret = pnl / entry_price - 2 * fee_rate
            trades.append(ret)
        if trades:
            series = pd.Series(trades)
            total_ret = (series + 1).prod() - 1
            sharpe = series.mean() / series.std() * np.sqrt(len(series)) if series.std() else np.nan
        else:
            total_ret = 0
            sharpe = np.nan
        results.append({'symbol': symbol, 'ret': total_ret, 'sharpe': sharpe})

    df_res = pd.DataFrame(results)
    return df_res['ret'].mean(), df_res['sharpe'].mean()


def main():
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql('SELECT * FROM features', engine, parse_dates=['open_time','close_time'])

    param_grid = {
        'history_window': [300, 500],
        'th_base': [0.10, 0.12],
        'ai_w': [0.15, 0.25],
        'trend_w': [0.2, 0.3],
    }
    grid = ParameterGrid(param_grid)
    best = None
    best_metric = -np.inf

    for params in grid:
        base_weights = {
            'ai': params['ai_w'],
            'trend': params['trend_w'],
            'momentum': 0.2,
            'volatility': 0.2,
            'volume': 0.1,
            'sentiment': 0.05,
            'funding': 0.05,
        }
        th_params = {'base': params['th_base'], 'min_thres': 0.06, 'max_thres': 0.25}
        tot_ret, sharpe = run_single_backtest(df, base_weights, params['history_window'], th_params)
        metric = sharpe if not np.isnan(sharpe) else -np.inf
        if metric > best_metric:
            best_metric = metric
            best = params
        print(f"params={params} -> total_ret={tot_ret:.4f}, sharpe={sharpe:.4f}")

    print("best params:", best, "best_sharpe:", best_metric)


if __name__ == '__main__':
    main()

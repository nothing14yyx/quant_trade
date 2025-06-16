import argparse
from collections import deque
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import optuna

from robust_signal_generator import RobustSignalGenerator
from backtester import (
    FEATURE_COLS_1H,
    FEATURE_COLS_4H,
    FEATURE_COLS_D1,
    MODEL_PATHS,
    load_config,
    connect_mysql,
    convert_model_paths,
)


def compute_ic_scores(df: pd.DataFrame, rsg: RobustSignalGenerator) -> dict:
    """根据历史数据计算各因子的 IC 分数"""
    df = df.sort_values("open_time").reset_index(drop=True)
    returns = df["close"].shift(-1) / df["open"].shift(-1) - 1
    scores = {k: [] for k in rsg.base_weights}

    for i in range(len(df) - 1):
        feats = {c: df.at[i, c] for c in FEATURE_COLS_1H if c in df}
        ai_score = rsg.get_ai_score(
            feats, rsg.models["1h"]["up"], rsg.models["1h"]["down"]
        )
        factors = rsg.get_factor_scores(feats, "1h")
        data = {"ai": ai_score, **factors}
        for k in scores:
            scores[k].append(data.get(k, np.nan))

    ic_scores = {}
    rets = returns.iloc[:-1].values
    thresh = 1e-12
    for k, vals in scores.items():
        arr = np.asarray(vals, dtype=float)
        mask = ~np.isnan(arr) & ~np.isnan(rets)
        if mask.sum() > 1:
            std_arr = arr[mask].std()
            std_rets = rets[mask].std()
            if std_arr < thresh or std_rets < thresh:
                ic = 0.0
            else:
                ic = np.corrcoef(arr[mask], rets[mask])[0, 1]
        else:
            ic = 0.0
        ic_scores[k] = ic
    return ic_scores


def precompute_ic_scores(df: pd.DataFrame, rsg: RobustSignalGenerator) -> dict:
    """预先计算一次 IC 分数，供多次回测复用"""
    return compute_ic_scores(df, rsg)


def run_single_backtest(
    df: pd.DataFrame,
    base_weights: dict,
    history_window: int,
    th_params: dict,
    ic_scores: dict,
    sg: RobustSignalGenerator,
) -> tuple:
    """在给定参数下执行一次回测，返回平均收益率和夏普比"""
    sg.history_scores = deque(maxlen=history_window)
    sg.base_weights = base_weights

    if th_params:
        def dyn_th(
            self,
            atr,
            adx,
            funding=0,
            atr_4h=None,
            adx_4h=None,
            atr_d1=None,
            adx_d1=None,
            pred_vol=None,
            pred_vol_4h=None,
            pred_vol_d1=None,
            vix_proxy=None,
            regime=None,
        ):
            return RobustSignalGenerator.dynamic_threshold(
                self,
                atr,
                adx,
                funding,
                atr_4h=atr_4h,
                adx_4h=adx_4h,
                atr_d1=atr_d1,
                adx_d1=adx_d1,
                pred_vol=pred_vol,
                pred_vol_4h=pred_vol_4h,
                pred_vol_d1=pred_vol_d1,
                vix_proxy=vix_proxy,
                base=th_params.get("base", 0.12409861615448753),
                regime=regime,
            )

        sg.dynamic_threshold = dyn_th.__get__(sg, RobustSignalGenerator)

    sg.ic_scores.update(ic_scores)

    all_symbols = df["symbol"].unique().tolist()
    fee_rate = 0.0005
    slippage = 0.0003
    results = []

    for symbol in all_symbols:
        df_sym = df[df["symbol"] == symbol].copy()
        df_sym = df_sym.sort_values("open_time").reset_index(drop=True)
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
            signals.append({
                "open_time": df_sym.at[i, "open_time"],
                "signal": res["signal"],
                "score": res["score"],
                "position_size": res.get("position_size", 1.0),
            })
        sig_df = pd.DataFrame(signals)
        valid_idx = sig_df[sig_df["signal"] != 0].index + 1
        valid_idx = valid_idx[valid_idx < len(df_sym)]
        trades = []
        for idx in valid_idx:
            entry_price = df_sym.at[idx, "open"] * (1 + slippage * np.sign(sig_df.at[idx - 1, "signal"]))
            exit_price = df_sym.at[idx, "close"] * (1 - slippage * np.sign(sig_df.at[idx - 1, "signal"]))
            direction = sig_df.at[idx - 1, "signal"]
            pos_size = sig_df.at[idx - 1, "position_size"]
            pnl = (exit_price - entry_price) * direction * pos_size
            if pos_size:
                ret = pnl / (entry_price * pos_size) - 2 * fee_rate
            else:
                ret = 0.0
            trades.append(ret)
        if trades:
            series = pd.Series(trades)
            total_ret = (series + 1).prod() - 1
            sharpe = series.mean() / series.std() * np.sqrt(len(series)) if series.std() else np.nan
        else:
            total_ret = 0
            sharpe = np.nan
        results.append({"symbol": symbol, "ret": total_ret, "sharpe": sharpe})

    df_res = pd.DataFrame(results)
    return df_res["ret"].mean(), df_res["sharpe"].mean()



def run_param_search(
    rows: int | None = None,
    method: str = "grid",
    trials: int = 30,
    tune_delta: bool = False,
) -> None:
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql("SELECT * FROM features", engine, parse_dates=["open_time", "close_time"])
    if rows:
        df = df.tail(rows)

    # 预先加载模型并实例化一次信号生成器
    sg = RobustSignalGenerator(
        model_paths=convert_model_paths(MODEL_PATHS),
        feature_cols_1h=FEATURE_COLS_1H,
        feature_cols_4h=FEATURE_COLS_4H,
        feature_cols_d1=FEATURE_COLS_D1,
    )
    cached_ic = precompute_ic_scores(df, sg)
    base_delta = sg.delta_params.copy()
    if method == "grid":
        param_grid = {
            "history_window": [300, 500],
            "th_base": [0.10, 0.12],
            "ai_w": [0.15, 0.25],
            "trend_w": [0.2, 0.3],
        }
        if tune_delta:
            param_grid.update({
                "rsi_inc": [0.03, 0.05],
                "macd_hist_inc": [0.03, 0.05],
                "ema_diff_inc": [0.02, 0.04],
                "atr_pct_inc": [0.02, 0.04],
                "vol_ma_ratio_inc": [0.02, 0.04],
                "funding_rate_inc": [0.02, 0.04],
            })
        grid = ParameterGrid(param_grid)
        best = None
        best_metric = -np.inf

        for params in tqdm(grid, desc="Grid Search"):
            base_weights = {
                "ai": params["ai_w"],
                "trend": params["trend_w"],
                "momentum": 0.2,
                "volatility": 0.2,
                "volume": 0.1,
                "sentiment": 0.05,
                "funding": 0.05,
            }
            th_params = {
                "base": params["th_base"],
            }
            delta_params = base_delta.copy()
            if tune_delta:
                delta_params["rsi"] = (
                    delta_params["rsi"][0],
                    delta_params["rsi"][1],
                    params["rsi_inc"],
                )
                delta_params["macd_hist"] = (
                    delta_params["macd_hist"][0],
                    delta_params["macd_hist"][1],
                    params["macd_hist_inc"],
                )
                delta_params["ema_diff"] = (
                    delta_params["ema_diff"][0],
                    delta_params["ema_diff"][1],
                    params["ema_diff_inc"],
                )
                delta_params["atr_pct"] = (
                    delta_params["atr_pct"][0],
                    delta_params["atr_pct"][1],
                    params["atr_pct_inc"],
                )
                delta_params["vol_ma_ratio"] = (
                    delta_params["vol_ma_ratio"][0],
                    delta_params["vol_ma_ratio"][1],
                    params["vol_ma_ratio_inc"],
                )
                delta_params["funding_rate"] = (
                    delta_params["funding_rate"][0],
                    delta_params["funding_rate"][1],
                    params["funding_rate_inc"],
                )
            sg_iter = RobustSignalGenerator(
                model_paths=convert_model_paths(MODEL_PATHS),
                feature_cols_1h=FEATURE_COLS_1H,
                feature_cols_4h=FEATURE_COLS_4H,
                feature_cols_d1=FEATURE_COLS_D1,
                delta_params=delta_params,
            )
            tot_ret, sharpe = run_single_backtest(
                df,
                base_weights,
                params["history_window"],
                th_params,
                cached_ic,
                sg_iter,
            )
            metric = sharpe if not np.isnan(sharpe) else -np.inf
            if metric > best_metric:
                best_metric = metric
                best = params
            print(
                f"params={params} -> total_ret={tot_ret:.4f}, sharpe={sharpe:.4f}"
            )

        print("best params:", best, "best_sharpe:", best_metric)
    else:
        def objective(trial: optuna.Trial) -> float:
            base_weights = {
                "ai": trial.suggest_float("ai_w", 0.1, 0.3),
                "trend": trial.suggest_float("trend_w", 0.1, 0.3),
                "momentum": trial.suggest_float("momentum_w", 0.1, 0.3),
                "volatility": 0.2,
                "volume": 0.1,
                "sentiment": 0.05,
                "funding": 0.05,
            }
            th_params = {
                "base": trial.suggest_float("th_base", 0.06, 0.15),
            }
            history_window = trial.suggest_int("history_window", 300, 800)
            delta_params = base_delta.copy()
            if tune_delta:
                delta_params["rsi"] = (
                    delta_params["rsi"][0],
                    delta_params["rsi"][1],
                    trial.suggest_float("rsi_inc", 0.02, 0.06),
                )
                delta_params["macd_hist"] = (
                    delta_params["macd_hist"][0],
                    delta_params["macd_hist"][1],
                    trial.suggest_float("macd_hist_inc", 0.02, 0.06),
                )
                delta_params["ema_diff"] = (
                    delta_params["ema_diff"][0],
                    delta_params["ema_diff"][1],
                    trial.suggest_float("ema_diff_inc", 0.02, 0.05),
                )
                delta_params["atr_pct"] = (
                    delta_params["atr_pct"][0],
                    delta_params["atr_pct"][1],
                    trial.suggest_float("atr_pct_inc", 0.02, 0.05),
                )
                delta_params["vol_ma_ratio"] = (
                    delta_params["vol_ma_ratio"][0],
                    delta_params["vol_ma_ratio"][1],
                    trial.suggest_float("vol_ma_ratio_inc", 0.02, 0.05),
                )
                delta_params["funding_rate"] = (
                    delta_params["funding_rate"][0],
                    delta_params["funding_rate"][1],
                    trial.suggest_float("funding_rate_inc", 0.02, 0.05),
                )
            sg_iter = RobustSignalGenerator(
                model_paths=convert_model_paths(MODEL_PATHS),
                feature_cols_1h=FEATURE_COLS_1H,
                feature_cols_4h=FEATURE_COLS_4H,
                feature_cols_d1=FEATURE_COLS_D1,
                delta_params=delta_params,
            )
            _, sharpe = run_single_backtest(
                df, base_weights, history_window, th_params, cached_ic, sg_iter
            )
            return sharpe if not np.isnan(sharpe) else -np.inf

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        print("best params:", study.best_params, "best_sharpe:", study.best_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000, help="只取最近 N 行数据")
    parser.add_argument(
        "--method",
        choices=["grid", "optuna"],
        default="optuna",
        help="搜索方式: grid 或 optuna",
    )
    parser.add_argument("--trials", type=int, default=30, help="Optuna 试验次数")
    parser.add_argument(
        "--tune-delta",
        action="store_true",
        help="同时搜索 Δ 参数增益",
    )
    args = parser.parse_args()
    run_param_search(
        rows=args.rows,
        method=args.method,
        trials=args.trials,
        tune_delta=args.tune_delta,
    )



if __name__ == "__main__":
    main()

import argparse
from collections import deque
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna

from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.backtester import (
    FEATURE_COLS_1H,
    FEATURE_COLS_4H,
    FEATURE_COLS_D1,
    MODEL_PATHS,
    load_config,
    connect_mysql,
    convert_model_paths,
    simulate_trades,
)


def compute_ic_scores(df: pd.DataFrame, rsg: RobustSignalGenerator) -> dict:
    """根据历史数据计算各因子的 IC 分数"""
    df = df.sort_values("open_time").reset_index(drop=True)
    returns = df["close"].shift(-1) / df["open"].shift(-1) - 1
    scores = {k: [] for k in rsg.base_weights}

    for i in range(len(df) - 1):
        feats = {c: df.at[i, c] for c in FEATURE_COLS_1H if c in df}
        models_1h = rsg.models.get("1h", {})
        if "up" in models_1h and "down" in models_1h:
            ai_score = rsg.get_ai_score(
                feats, models_1h["up"], models_1h["down"]
            )
        elif "cls" in models_1h:
            ai_score = rsg.get_ai_score_cls(feats, models_1h["cls"])
        else:
            ai_score = 0.0
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
            base=None,
            low_base=None,
            reversal=False,
            history_scores=None,
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
                base=th_params.get("base", base),
                regime=regime,
                low_base=low_base,
                reversal=reversal,
                history_scores=history_scores,
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
                "take_profit": res.get("take_profit"),
                "stop_loss": res.get("stop_loss"),
            })
        sig_df = pd.DataFrame(signals)
        trades_df = simulate_trades(
            df_sym, sig_df, fee_rate=fee_rate, slippage=slippage
        )
        if trades_df.empty:
            total_ret = 0
            sharpe = 0.0
        else:
            series = trades_df["ret"]
            std = series.std()
            total_ret = (series + 1).prod() - 1
            sharpe = 0.0 if std == 0 else series.mean() / std * np.sqrt(len(series))
        results.append({"symbol": symbol, "ret": total_ret, "sharpe": sharpe})

    df_res = pd.DataFrame(results)
    return df_res["ret"].mean(), df_res["sharpe"].mean()



def run_param_search(
    rows: int | None = None,
    method: str = "grid",
    trials: int = 30,
    tune_delta: bool = False,
    n_jobs: int = 1,
    test_ratio: float = 0.2,
) -> None:
    """Search for optimal parameters using a train/validation split.

    Parameters
    ----------
    rows: int | None
        Number of recent rows to use from the features table.
    method: str
        Search algorithm: "grid" or "optuna".
    trials: int
        Number of trials when using optuna.
    tune_delta: bool
        Whether to tune delta boost parameters.
    n_jobs: int
        Parallel jobs for grid search.
    test_ratio: float
        Portion of data used as validation set.
    """
    cfg = load_config()
    engine = connect_mysql(cfg)
    df = pd.read_sql(
        "SELECT * FROM features", engine, parse_dates=["open_time", "close_time"]
    )
    if df.empty:
        raise ValueError("features 表无数据")
    if rows:
        df = df.tail(rows)
    df = df.sort_values("open_time").reset_index(drop=True)
    train_len = int(len(df) * (1 - test_ratio))
    train_df, valid_df = df.iloc[:train_len], df.iloc[train_len:]

    # 预先加载模型并实例化一次信号生成器
    sg = RobustSignalGenerator(
        model_paths=convert_model_paths(MODEL_PATHS),
        feature_cols_1h=FEATURE_COLS_1H,
        feature_cols_4h=FEATURE_COLS_4H,
        feature_cols_d1=FEATURE_COLS_D1,
    )
    cached_ic = precompute_ic_scores(train_df, sg)
    base_delta = sg.delta_params.copy()
    if method == "grid":
        param_grid = {
            "history_window": [300, 500],
            "th_base": [0.10, 0.12],
            "ai_w": [0.1, 0.3],
            "trend_w": [0.1, 0.3],
            "momentum_w": [0.1, 0.3],
            "volatility_w": [0.1, 0.3],
            "volume_w": [0.05, 0.2],
            "sentiment_w": [0.05, 0.2],
            "funding_w": [0.05, 0.2],
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
        grid = list(ParameterGrid(param_grid))

        def eval_params(params: dict) -> tuple[float, dict, float, float]:
            def _get(v):
                return v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
            keys = [
                "ai",
                "trend",
                "momentum",
                "volatility",
                "volume",
                "sentiment",
                "funding",
            ]
            weights = np.array(
                [
                    _get(params["ai_w"]),
                    _get(params["trend_w"]),
                    _get(params["momentum_w"]),
                    _get(params["volatility_w"]),
                    _get(params["volume_w"]),
                    _get(params["sentiment_w"]),
                    _get(params["funding_w"]),
                ]
            )
            weights /= weights.sum() if weights.sum() != 0 else 1.0
            base_weights = dict(zip(keys, weights))
            th_params = {"base": _get(params["th_base"])}
            delta_params = base_delta.copy()
            if tune_delta:
                delta_params["rsi"] = (
                    delta_params["rsi"][0],
                    delta_params["rsi"][1],
                    _get(params["rsi_inc"]),
                )
                delta_params["macd_hist"] = (
                    delta_params["macd_hist"][0],
                    delta_params["macd_hist"][1],
                    _get(params["macd_hist_inc"]),
                )
                delta_params["ema_diff"] = (
                    delta_params["ema_diff"][0],
                    delta_params["ema_diff"][1],
                    _get(params["ema_diff_inc"]),
                )
                delta_params["atr_pct"] = (
                    delta_params["atr_pct"][0],
                    delta_params["atr_pct"][1],
                    _get(params["atr_pct_inc"]),
                )
                delta_params["vol_ma_ratio"] = (
                    delta_params["vol_ma_ratio"][0],
                    delta_params["vol_ma_ratio"][1],
                    _get(params["vol_ma_ratio_inc"]),
                )
                delta_params["funding_rate"] = (
                    delta_params["funding_rate"][0],
                    delta_params["funding_rate"][1],
                    _get(params["funding_rate_inc"]),
                )
            sg_iter = RobustSignalGenerator(
                model_paths=convert_model_paths(MODEL_PATHS),
                feature_cols_1h=FEATURE_COLS_1H,
                feature_cols_4h=FEATURE_COLS_4H,
                feature_cols_d1=FEATURE_COLS_D1,
                delta_params=delta_params,
            )
            tot_ret, sharpe = run_single_backtest(
                valid_df,
                base_weights,
                _get(params["history_window"]),
                th_params,
                cached_ic,
                sg_iter,
            )
            metric = sharpe if not np.isnan(sharpe) else -np.inf
            return metric, params, tot_ret, sharpe

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(eval_params)(p) for p in tqdm(grid, desc="Grid Search")
        )

        best = None
        best_metric = -np.inf
        for metric, params, tot_ret, sharpe in results:
            if metric > best_metric:
                best_metric = metric
                best = params
            print(
                f"params={params} -> total_ret={tot_ret:.4f}, sharpe={sharpe:.4f}"
            )

        print("best params:", best, "best_sharpe:", best_metric)
    else:
        def objective(trial: optuna.Trial) -> float:
            keys = [
                "ai",
                "trend",
                "momentum",
                "volatility",
                "volume",
                "sentiment",
                "funding",
            ]
            weights = np.array(
                [
                    trial.suggest_float("ai_w", 0.1, 0.3),
                    trial.suggest_float("trend_w", 0.1, 0.3),
                    trial.suggest_float("momentum_w", 0.1, 0.3),
                    trial.suggest_float("volatility_w", 0.1, 0.3),
                    trial.suggest_float("volume_w", 0.05, 0.2),
                    trial.suggest_float("sentiment_w", 0.05, 0.2),
                    trial.suggest_float("funding_w", 0.05, 0.2),
                ]
            )
            weights /= weights.sum() if weights.sum() != 0 else 1.0
            base_weights = dict(zip(keys, weights))
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
                valid_df,
                base_weights,
                history_window,
                th_params,
                cached_ic,
                sg_iter,
            )
            return sharpe if not np.isnan(sharpe) else -np.inf

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        print("best params:", study.best_params, "best_sharpe:", study.best_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000, help="只取最近 N 行数据")
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="并行搜索的任务数，仅对 grid 模式有效",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="验证集所占比例",
    )
    args = parser.parse_args()
    run_param_search(
        rows=args.rows,
        method=args.method,
        trials=args.trials,
        tune_delta=args.tune_delta,
        n_jobs=args.n_jobs,
        test_ratio=args.test_ratio,
    )



if __name__ == "__main__":
    main()

import argparse
from collections import deque
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna

from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
    DynamicThresholdInput,

)
from quant_trade.backtester import (
    FEATURE_COLS_1H,
    FEATURE_COLS_4H,
    FEATURE_COLS_D1,
    MODEL_PATHS,
    simulate_trades,
)
from quant_trade.utils.db import load_config, connect_mysql

logger = logging.getLogger(__name__)


def compute_ic_scores(df: pd.DataFrame, rsg: RobustSignalGenerator) -> dict:
    """根据历史数据计算各因子的 IC 分数"""
    df = df.sort_values("open_time").reset_index(drop=True)
    returns = df["close"].shift(-1) / df["open"].shift(-1) - 1
    scores = {k: [] for k in rsg.base_weights}

    for i in range(len(df) - 1):
        feats = {c: df.at[i, c] for c in FEATURE_COLS_1H if c in df}
        models_1h = rsg.models.get("1h", {})
        if "up" in models_1h and "down" in models_1h:
            ai_score = rsg.predictor.get_ai_score(
                feats, models_1h["up"], models_1h["down"]
            )
        elif "cls" in models_1h:
            ai_score = rsg.predictor.get_ai_score_cls(feats, models_1h["cls"])
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
    """在给定参数下执行一次回测，返回平均收益率、夏普比和交易笔数"""
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
            **kwargs,
        ):
            data = DynamicThresholdInput(
                atr=atr,
                adx=adx,
                bb_width_chg=kwargs.get("bb_width_chg"),
                channel_pos=kwargs.get("channel_pos"),
                funding=funding,
                atr_4h=atr_4h,
                adx_4h=adx_4h,
                atr_d1=atr_d1,
                adx_d1=adx_d1,
                pred_vol=pred_vol,
                pred_vol_4h=pred_vol_4h,
                pred_vol_d1=pred_vol_d1,
                vix_proxy=vix_proxy,
                regime=regime,
                reversal=reversal,
            )
            return RobustSignalGenerator.compute_dynamic_threshold(
                self,
                data,
                base=th_params.get("base", base),
                low_base=low_base,
                history_scores=history_scores,
            )

        sg.dynamic_threshold = dyn_th.__get__(sg, RobustSignalGenerator)

    sg.ic_scores.update(ic_scores)
    cfg = load_config()
    all_symbols = df["symbol"].unique().tolist()
    costs_cfg = cfg.get("costs", {})
    fee_rate = float(costs_cfg.get("fee_rate", 0.0005))
    slippage = float(costs_cfg.get("slippage", 0.0003))
    results = []
    port_rets: list[float] = []
    total_trades = 0

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
            if res is None:
                res = {
                    "signal": 0,
                    "score": float("nan"),
                    "position_size": 0.0,
                    "take_profit": None,
                    "stop_loss": None,
                }
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
        # === Debug & 清洗 NaN BEGIN ===
        nan_ret = trades_df["ret"].isna().sum()
        nan_pos = trades_df["position_size"].isna().sum()
        if nan_ret or nan_pos:
            logger.warning(
                "[NaN DETECT] %s  trades=%d  NaN_ret=%d  NaN_pos=%d",
                symbol, len(trades_df), nan_ret, nan_pos,
            )

        trades_df = trades_df.dropna(subset=["ret", "position_size"])
        trade_count = len(trades_df)
        if trade_count == 0:
            logger.debug("%s -> no valid trades after cleaning", symbol)
            continue
        total_trades += trade_count
        # === Debug & 清洗 NaN END ===
        series = trades_df["ret"] * trades_df["position_size"]
        if trade_count < 2:
            logger.debug("%s -> trades=%d < 2, sharpe may be NaN", symbol, trade_count)
        std = series.std()
        logger.debug(
            "%s -> trade_count=%d mean=%.6f std=%.6f", symbol, trade_count, series.mean(), std
        )
        total_ret = (series + 1).prod() - 1
        symbol_sharpe = (
            series.mean() / std * np.sqrt(len(series))
            if trade_count >= 2 and std != 0
            else np.nan
        )
        port_rets.extend(series.tolist())
        results.append({"symbol": symbol, "ret": total_ret, "sharpe": symbol_sharpe})

    port_rets = np.asarray(port_rets, dtype=float)
    if port_rets.size == 0:
        logger.warning("no trades generated for current parameters")
    if len(port_rets) >= 2 and port_rets.std() != 0:
        sharpe = port_rets.mean() / port_rets.std() * np.sqrt(len(port_rets))
    else:
        sharpe = np.nan
    mean_ret = port_rets.mean() if len(port_rets) > 0 else np.nan
    return mean_ret, sharpe, len(port_rets)



def run_param_search(
    rows: int | None = None,
    method: str = "optuna",
    trials: int = 30,
    tune_delta: bool = True,
    n_jobs: int = 1,
    test_ratio: float = 0.2,
    n_splits: int = 1,
    start_time: pd.Timestamp | str | None = None,
) -> None:
    """Search for optimal parameters using a train/validation split.

    Parameters
    ----------
    rows: int | None
        Number of recent rows to query from the features table.
    start_time: pandas.Timestamp or str, optional
        Only load data with ``open_time`` greater than or equal to this value.
    method: str
        Search algorithm: "grid" or "optuna".
    trials: int
        Number of trials when using optuna.
    tune_delta: bool
        Whether to tune delta boost parameters. Defaults to ``True``.
    n_jobs: int
        Parallel jobs for grid search.
    test_ratio: float
        Portion of data used as validation set.
    n_splits: int
        Number of CV folds. When greater than 1, ``TimeSeriesSplit`` is used
        to generate train/validation splits and the average Sharpe ratio is
        evaluated.
    """
    cfg = load_config()
    engine = connect_mysql(cfg)
    query = "SELECT * FROM features"
    params: dict[str, object] | None = None
    conds = []
    if start_time is not None:
        conds.append("open_time >= %(start)s")
        params = {"start": pd.to_datetime(start_time)}
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY open_time DESC"
    if rows:
        query += f" LIMIT {rows}"
    df = pd.read_sql(
        query,
        engine,
        params=params,
        parse_dates=["open_time", "close_time"],
    )
    if df.empty:
        raise ValueError("features 表无数据")
    df = df.sort_values("open_time").reset_index(drop=True)
    if n_splits <= 1:
        train_len = int(len(df) * (1 - test_ratio))
        splits = [(df.iloc[:train_len], df.iloc[train_len:])]
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = [
            (df.iloc[tr_idx], df.iloc[val_idx])
            for tr_idx, val_idx in tscv.split(df)
        ]

    cfg = load_config()
    rsg_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
    sg = RobustSignalGenerator(rsg_cfg)
    cached_ics = [precompute_ic_scores(tr, sg) for tr, _ in splits]
    base_delta = sg.delta_params.copy()
    if method == "grid":
        param_grid = {
            "history_window": [300, 500],
            "th_base": [0.03, 0.08],
            "ai_w": [0.05, 0.35],
            "trend_w": [0.05, 0.35],
            "momentum_w": [0.05, 0.35],
            "volatility_w": [0.05, 0.35],
            "volume_w": [0.05, 0.35],
            "sentiment_w": [0.05, 0.35],
            "funding_w": [0.05, 0.35],
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
            ret_vals = []
            sharpe_vals = []
            trades = 0
            for (tr_df, val_df), ic in zip(splits, cached_ics):
                cfg = load_config()
                iter_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
                iter_cfg.delta_params = delta_params
                sg_iter = RobustSignalGenerator(iter_cfg)
                tot_ret, sharpe, trade_count = run_single_backtest(
                    val_df,
                    base_weights,
                    _get(params["history_window"]),
                    th_params,
                    ic,
                    sg_iter,
                )
                if hasattr(sg_iter, "stop_weight_update_thread"):
                    sg_iter.stop_weight_update_thread()
                ret_vals.append(tot_ret)
                sharpe_vals.append(sharpe)
                trades += trade_count
            ret_arr = np.asarray(ret_vals, dtype=float)
            sharpe_arr = np.asarray(sharpe_vals, dtype=float)
            if ret_arr.size == 0 or np.isnan(ret_arr).all():
                mean_ret = np.nan
            else:
                mean_ret = float(np.nanmean(ret_arr))
            if sharpe_arr.size == 0 or np.isnan(sharpe_arr).all():
                mean_sharpe = np.nan
            else:
                mean_sharpe = float(np.nanmean(sharpe_arr))
            metric = mean_sharpe if not np.isnan(mean_sharpe) else -100.0
            return metric, params, mean_ret, mean_sharpe, trades

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(eval_params)(p) for p in tqdm(grid, desc="Grid Search")
        )

        if all(r[4] == 0 for r in results):
            raise ValueError("no trades found during parameter search")

        best = None
        best_metric = -np.inf
        for metric, params, tot_ret, sharpe, trade_count in results:
            if metric > best_metric:
                best_metric = metric
                best = params
            logger.info(
                "params=%s -> trades=%d total_ret=%.4f, sharpe=%.6f",
                params,
                trade_count,
                tot_ret,
                sharpe,
            )

        logger.info("best params: %s best_sharpe: %.6f", best, best_metric)
        if hasattr(sg, "stop_weight_update_thread"):
            sg.stop_weight_update_thread()
        return best_metric
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
                    trial.suggest_float("ai_w", 0.05, 0.35),
                    trial.suggest_float("trend_w", 0.05, 0.35),
                    trial.suggest_float("momentum_w", 0.05, 0.35),
                    trial.suggest_float("volatility_w", 0.05, 0.35),
                    trial.suggest_float("volume_w", 0.05, 0.35),
                    trial.suggest_float("sentiment_w", 0.05, 0.35),
                    trial.suggest_float("funding_w", 0.05, 0.35),
                ]
            )
            weights /= weights.sum() if weights.sum() != 0 else 1.0
            base_weights = dict(zip(keys, weights))
            th_params = {
                "base": trial.suggest_float("th_base", 0.03, 0.10),
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
            ret_vals = []
            sharpe_vals = []
            trades = 0
            for (tr_df, val_df), ic in zip(splits, cached_ics):
                cfg = load_config()
                iter_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
                iter_cfg.delta_params = delta_params
                sg_iter = RobustSignalGenerator(iter_cfg)
                tot_ret, sharpe, trade_count = run_single_backtest(
                    val_df,
                    base_weights,
                    history_window,
                    th_params,
                    ic,
                    sg_iter,
                )
                if hasattr(sg_iter, "stop_weight_update_thread"):
                    sg_iter.stop_weight_update_thread()
                ret_vals.append(tot_ret)
                sharpe_vals.append(sharpe)
                trades += trade_count
            ret_arr = np.asarray(ret_vals, dtype=float)
            sharpe_arr = np.asarray(sharpe_vals, dtype=float)
            if ret_arr.size == 0 or np.isnan(ret_arr).all():
                mean_ret = np.nan
            else:
                mean_ret = float(np.nanmean(ret_arr))
            if sharpe_arr.size == 0 or np.isnan(sharpe_arr).all():
                mean_sharpe = np.nan
            else:
                mean_sharpe = float(np.nanmean(sharpe_arr))
            logger.debug(
                "optuna params=%s -> trades=%d total_ret=%.4f, sharpe=%.6f",
                {
                    "ai_w": weights[0],
                    "trend_w": weights[1],
                    "momentum_w": weights[2],
                    "volatility_w": weights[3],
                    "volume_w": weights[4],
                    "sentiment_w": weights[5],
                    "funding_w": weights[6],
                    "th_base": th_params["base"],
                    "history_window": history_window,
                    **(
                        {
                            "rsi_inc": delta_params["rsi"][2],
                            "macd_hist_inc": delta_params["macd_hist"][2],
                            "ema_diff_inc": delta_params["ema_diff"][2],
                            "atr_pct_inc": delta_params["atr_pct"][2],
                            "vol_ma_ratio_inc": delta_params["vol_ma_ratio"][2],
                            "funding_rate_inc": delta_params["funding_rate"][2],
                        }
                        if tune_delta
                        else {}
                    ),
                },
                trades,
                mean_ret,
                mean_sharpe,
            )
            trial.set_user_attr("trades", trades)
            if np.isnan(mean_sharpe):
                return -100.0  # 给可比较的负分，避免 -inf
            return mean_sharpe
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=True)

        if all(t.user_attrs.get("trades", 0) == 0 for t in study.trials):
            raise ValueError("no trades found during parameter search")

        logger.info(
            "best params: %s best_sharpe: %.6f", study.best_params, study.best_value
        )
        if hasattr(sg, "stop_weight_update_thread"):
            sg.stop_weight_update_thread()
        return study.best_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000, help="只取最近 N 行数据")
    parser.add_argument(
        "--method",
        choices=["grid", "optuna"],
        default="optuna",
        help="搜索方式: grid 或 optuna",
    )
    parser.add_argument("--trials", type=int, default=30, help="Optuna 试验次数")
    parser.add_argument(
        "--tune-delta",
        dest="tune_delta",
        action="store_true",
        help="同时搜索 Δ 参数增益",
    )
    parser.add_argument(
        "--no-tune-delta",
        dest="tune_delta",
        action="store_false",
        help="关闭 Δ 参数增益搜索",
    )
    parser.set_defaults(tune_delta=True)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="并行搜索的任务数，仅对 grid 模式有效",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.4,
        help="验证集所占比例",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="交叉验证折数，大于1时启用 TimeSeriesSplit",
    )
    args = parser.parse_args()
    run_param_search(
        rows=args.rows,
        method=args.method,
        trials=args.trials,
        tune_delta=args.tune_delta,
        n_jobs=args.n_jobs,
        test_ratio=args.test_ratio,
        n_splits=args.cv_folds,
    )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

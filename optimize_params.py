import argparse
from pathlib import Path
import yaml
import pandas as pd
import optuna

from quant_trade.utils.db import load_config, connect_mysql, CONFIG_PATH
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.param_search import precompute_ic_scores, run_single_backtest


KEYS = ["ai", "trend", "momentum", "volatility", "volume", "sentiment", "funding"]


def optimize_params(rows: int = 50000, trials: int = 30, out_path: Path | str = CONFIG_PATH):
    """使用 Optuna 搜索最佳参数并写入配置文件."""
    cfg = load_config()
    engine = connect_mysql(cfg)
    query = "SELECT * FROM features ORDER BY open_time DESC"
    if rows:
        query += f" LIMIT {rows}"
    df = pd.read_sql(query, engine, parse_dates=["open_time", "close_time"])
    if df.empty:
        raise ValueError("features 表无数据")
    df = df.sort_values("open_time").reset_index(drop=True)

    rsg_cfg = RobustSignalGeneratorConfig.from_cfg(cfg)
    sg = RobustSignalGenerator(rsg_cfg)
    ic_scores = precompute_ic_scores(df, sg)
    history_window = cfg.get("history_window", 679)

    def objective(trial: optuna.Trial) -> float:
        weights = [trial.suggest_float(f"{k}_w", 0.05, 0.3) for k in KEYS]
        tot = sum(weights) or 1.0
        base_weights = {k: w / tot for k, w in zip(KEYS, weights)}
        base_th = trial.suggest_float("base_th", 0.05, 0.15)
        risk_adj = trial.suggest_float("risk_adjust_factor", 0.1, 0.9)
        sg.base_weights = base_weights
        sg.risk_adjust_factor = risk_adj
        _, sharpe, trades = run_single_backtest(
            df, base_weights, history_window, {"base": base_th}, ic_scores, sg
        )
        if trades == 0 or pd.isna(sharpe):
            return -100.0
        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_params

    cfg.setdefault("signal_threshold", {})["base_th"] = float(best["base_th"])
    if "ic_scores" in cfg and "base_weights" in cfg["ic_scores"]:
        tot = sum(best[f"{k}_w"] for k in KEYS) or 1.0
        for k in KEYS:
            cfg["ic_scores"]["base_weights"][k] = float(best[f"{k}_w"]) / tot
    cfg.setdefault("risk_adjust", {})["factor"] = float(best["risk_adjust_factor"])

    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000, help="读取最近 N 行数据")
    parser.add_argument("--trials", type=int, default=30, help="Optuna 试验次数")
    parser.add_argument("--output", default=str(CONFIG_PATH), help="结果配置文件路径")
    args = parser.parse_args()
    best = optimize_params(rows=args.rows, trials=args.trials, out_path=args.output)
    print("best params:", best)


if __name__ == "__main__":
    main()

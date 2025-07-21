import argparse
from pathlib import Path
import yaml
import pandas as pd
import optuna
import random

from quant_trade.utils.db import load_config, connect_mysql, CONFIG_PATH
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.param_search import precompute_ic_scores, run_single_backtest


KEYS = ["ai", "trend", "momentum", "volatility", "volume", "sentiment", "funding"]


def optimize_params(
    rows: int = 50000,
    trials: int = 30,
    out_path: Path | str = CONFIG_PATH,
    method: str = "optuna",
) -> dict:
    """搜索最佳参数并写入配置文件.

    Parameters
    ----------
    rows: int
        读取最近 N 行数据.
    trials: int
        迭代次数或试验次数.
    out_path: Path | str
        输出配置文件路径.
    method: str
        搜索方法, ``optuna`` 或 ``ga``.
    """
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

    def objective(trial_params) -> float:
        if hasattr(trial_params, "suggest_float"):
            weights = [trial_params.suggest_float(f"{k}_w", 0.05, 0.3) for k in KEYS]
            base_th = trial_params.suggest_float("base_th", 0.05, 0.15)
            risk_adj = trial_params.suggest_float("risk_adjust_factor", 0.1, 0.9)
        else:
            weights = trial_params["weights"]
            base_th = trial_params["base_th"]
            risk_adj = trial_params["risk_adj"]

        tot = sum(weights) or 1.0
        base_weights = {k: w / tot for k, w in zip(KEYS, weights)}
        sg.base_weights = base_weights
        sg.risk_adjust_factor = risk_adj
        _, sharpe, trades = run_single_backtest(
            df, base_weights, history_window, {"base": base_th}, ic_scores, sg
        )
        if trades == 0 or pd.isna(sharpe):
            return -100.0
        return sharpe

    def ga_search() -> dict:
        pop_size = max(4, trials)
        population = [
            {
                "weights": [random.uniform(0.05, 0.3) for _ in KEYS],
                "base_th": random.uniform(0.05, 0.15),
                "risk_adj": random.uniform(0.1, 0.9),
            }
            for _ in range(pop_size)
        ]
        best = None
        best_score = -float("inf")

        def mutate(ind, rate=0.1):
            for i in range(len(ind["weights"])):
                if random.random() < rate:
                    ind["weights"][i] += random.gauss(0, 0.02)
                    ind["weights"][i] = float(min(max(ind["weights"][i], 0.05), 0.3))
            if random.random() < rate:
                ind["base_th"] += random.gauss(0, 0.01)
                ind["base_th"] = float(min(max(ind["base_th"], 0.05), 0.15))
            if random.random() < rate:
                ind["risk_adj"] += random.gauss(0, 0.05)
                ind["risk_adj"] = float(min(max(ind["risk_adj"], 0.1), 0.9))

        for _ in range(trials):
            scores = [objective(ind) for ind in population]
            for ind, score in zip(population, scores):
                if score > best_score:
                    best = ind.copy()
                    best_score = score
            ranked = [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
            survivors = ranked[: max(2, pop_size // 2)]
            new_pop = survivors.copy()
            while len(new_pop) < pop_size:
                a, b = random.sample(survivors, 2)
                cut = random.randint(1, len(KEYS) - 1)
                child_w = a["weights"][:cut] + b["weights"][cut:]
                child = {
                    "weights": child_w,
                    "base_th": random.choice([a["base_th"], b["base_th"]]),
                    "risk_adj": random.choice([a["risk_adj"], b["risk_adj"]]),
                }
                mutate(child)
                new_pop.append(child)
            population = new_pop
        return best or population[0]

    if method == "ga":
        best_params = ga_search()
        best = {
            "base_th": best_params["base_th"],
            "risk_adjust_factor": best_params["risk_adj"],
        }
        for w, k in zip(best_params["weights"], KEYS):
            best[f"{k}_w"] = w
    else:
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
    parser.add_argument("--trials", type=int, default=30, help="搜索迭代次数")
    parser.add_argument(
        "--method",
        choices=["optuna", "ga"],
        default="optuna",
        help="搜索算法: optuna 或 ga",
    )
    parser.add_argument("--output", default=str(CONFIG_PATH), help="结果配置文件路径")
    args = parser.parse_args()
    best = optimize_params(
        rows=args.rows,
        trials=args.trials,
        out_path=args.output,
        method=args.method,
    )
    print("best params:", best)


if __name__ == "__main__":
    main()

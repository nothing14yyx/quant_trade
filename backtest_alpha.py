"""对比不同 CVaR alpha 下策略表现的简单回测脚本"""

from pathlib import Path
import pandas as pd
from quant_trade.backtester import run_backtest

ALPHAS = [0.05, 0.1, 0.2]


def main():
    results = []
    for alpha in ALPHAS:
        print(f"Running backtest with alpha={alpha}")
        run_backtest(cvar_alpha=alpha)
        summary_path = Path(__file__).resolve().parent / "backtest_fusion_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            total_ret = df["total_ret"].sum()
            sharpe = df["sharpe"].mean()
            trades = df["n_trades"].sum()
            results.append(
                {
                    "alpha": alpha,
                    "total_ret": total_ret,
                    "sharpe": sharpe,
                    "trades": trades,
                }
            )
    if results:
        print(pd.DataFrame(results))


if __name__ == "__main__":
    main()

import argparse
import difflib
import os
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

from quant_trade.signal.core import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.risk_manager import RiskManager

CONFIG_PATH = Path(__file__).resolve().parents[1] / "quant_trade" / "utils" / "config.yaml"
MIN_TRADES = 30


def _load_features(path: str) -> pd.DataFrame:
    """加载历史特征数据."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"features file not found: {file}")
    if file.suffix == ".parquet":
        return pd.read_parquet(file)
    return pd.read_csv(file)


def _extract_return_col(df: pd.DataFrame) -> str:
    candidates = [
        "target",
        "target_1h",
        "return",
        "ret",
        "pnl",
        "future_return",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("未找到 PnL 代理列，可在数据中提供 target/return 等字段")


def _grid(val: float) -> Iterable[float]:
    """围绕 val ±10% 构建三点网格."""
    return [val * 0.9, val, val * 1.1]


def _prepare_features(df: pd.DataFrame) -> Tuple[list, list, list]:
    cols_1h = [c for c in df.columns if c.endswith("_1h")]
    cols_4h = [c for c in df.columns if c.endswith("_4h")]
    cols_d1 = [c for c in df.columns if c.endswith("_d1")]
    feats_1h, feats_4h, feats_d1 = [], [], []
    for _, row in df.iterrows():
        feats_1h.append({c: row[c] for c in cols_1h})
        feats_4h.append({c: row[c] for c in cols_4h})
        feats_d1.append({c: row[c] for c in cols_d1})
    return feats_1h, feats_4h, feats_d1


def _evaluate(
    rsg_cfg: RobustSignalGeneratorConfig,
    feats_1h: list,
    feats_4h: list,
    feats_d1: list,
    returns: pd.Series,
    params: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """使用给定参数评估指标并返回综合得分."""
    sg = RobustSignalGenerator(rsg_cfg)
    sg.risk_manager = RiskManager()
    sg.ai_dir_eps = params["ai_dir_eps"]
    sg.signal_threshold_cfg["base_th"] = params["base_th"]
    sg.signal_threshold_cfg["dynamic_quantile"] = params["dynamic_quantile"]
    sg.signal_threshold_cfg["rev_boost"] = params["rev_boost"]
    sg.th_decay = params["th_decay"]

    records = []
    for f1, f4, fd, ret in zip(feats_1h, feats_4h, feats_d1, returns):
        res = sg.generate_signal(f1, f4, fd)
        diag = sg.diagnose()
        fused = diag.get("fused_score")
        base = diag.get("base_th")
        regime = diag.get("regime")
        signal = res.get("signal", 0.0) if res else 0.0
        pnl = signal * ret
        records.append(
            {
                "fused_score": fused,
                "base_th": base,
                "regime": regime,
                "signal": signal,
                "pnl": pnl,
            }
        )

    sg.update_weights()
    df_rec = pd.DataFrame(records)
    trades = df_rec[df_rec["signal"] != 0]
    if len(trades) < MIN_TRADES or df_rec["pnl"].std(ddof=0) == 0:
        return -np.inf, {}

    pnl = df_rec["pnl"]
    sharpe = pnl.mean() / pnl.std(ddof=0)
    downside = pnl[pnl < 0]
    sortino = (
        pnl.mean() / downside.std(ddof=0)
        if len(downside) > 0 and downside.std(ddof=0) != 0
        else 0.0
    )
    hit_rate = (pnl > 0).mean()
    score = 0.6 * sharpe + 0.2 * sortino + 0.2 * hit_rate
    return score, {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "hit_rate": float(hit_rate),
        "trades": int(len(trades)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate threshold parameters")
    parser.add_argument("--data", help="历史特征 parquet/CSV 路径")
    args = parser.parse_args()

    data_path = args.data or os.getenv("HIST_FEATURES_PATH")
    if not data_path:
        raise SystemExit("请通过 --data 或环境变量 HIST_FEATURES_PATH 指定特征文件路径")

    df = _load_features(data_path)
    ret_col = _extract_return_col(df)
    feats_1h, feats_4h, feats_d1 = _prepare_features(df)
    returns = df[ret_col]

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults = {
        "ai_dir_eps": cfg.get("vote_system", {}).get("ai_dir_eps", 0.1),
        "base_th": cfg.get("signal_threshold", {}).get("base_th", 0.1),
        "dynamic_quantile": cfg.get("signal_threshold", {}).get("dynamic_quantile", 0.8),
        "th_decay": cfg.get("th_decay", 1.0),
        "rev_boost": cfg.get("signal_threshold", {}).get("rev_boost", 0.15),
    }

    rsg_cfg = RobustSignalGeneratorConfig.from_file(CONFIG_PATH)

    grid = {
        k: _grid(v) for k, v in defaults.items()
    }

    best_score = -np.inf
    best_params: Dict[str, float] = {}
    best_metrics: Dict[str, float] = {}
    for combo in product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        score, metrics = _evaluate(
            rsg_cfg, feats_1h, feats_4h, feats_d1, returns, params
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    delta = {
        "ai_dir_eps": best_params["ai_dir_eps"] - defaults["ai_dir_eps"],
        "th_decay": best_params["th_decay"] - defaults["th_decay"],
        "signal_threshold": {
            "base_th": best_params["base_th"] - defaults["base_th"],
            "dynamic_quantile": best_params["dynamic_quantile"]
            - defaults["dynamic_quantile"],
            "rev_boost": best_params["rev_boost"] - defaults["rev_boost"],
        },
    }
    with open("calibrate_thresholds_delta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(delta, f, allow_unicode=True, sort_keys=False)

    report = ["# 阈值校准报告", "", "## 最佳参数", ""]
    for k, v in best_params.items():
        report.append(f"- **{k}**: {v:.6g}")
    report.extend(
        [
            "",
            "## 指标统计",
            f"- Sharpe: {best_metrics.get('sharpe', 0):.4f}",
            f"- Sortino: {best_metrics.get('sortino', 0):.4f}",
            f"- 命中率: {best_metrics.get('hit_rate', 0):.4f}",
            f"- 交易次数: {best_metrics.get('trades', 0)}",
        ]
    )
    with open("calibrate_thresholds_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # 生成拟议 diff
    new_cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    new_cfg.setdefault("vote_system", {})["ai_dir_eps"] = best_params["ai_dir_eps"]
    st = new_cfg.setdefault("signal_threshold", {})
    st["base_th"] = best_params["base_th"]
    st["dynamic_quantile"] = best_params["dynamic_quantile"]
    st["rev_boost"] = best_params["rev_boost"]
    new_cfg["th_decay"] = best_params["th_decay"]

    orig_text = Path(CONFIG_PATH).read_text(encoding="utf-8").splitlines(True)
    new_text = yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False).splitlines(
        True
    )
    diff = "".join(
        difflib.unified_diff(orig_text, new_text, fromfile="a/config.yaml", tofile="b/config.yaml")
    )
    print("\n拟议 diff:\n" + diff)


if __name__ == "__main__":
    main()

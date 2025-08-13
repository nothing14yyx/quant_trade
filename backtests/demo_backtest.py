import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    """读取模型与 report.json，执行简单推理与回测示例"""
    report_path = Path("models") / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("请先运行训练脚本生成模型和 report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if not report:
        raise ValueError("report.json 为空")

    model_info = next(iter(report.values()))
    bundle = joblib.load(model_info["path"])
    pipe = bundle["pipeline"]
    features = bundle["features"]
    threshold = bundle.get("threshold", 0.5)

    # 随机生成示例特征与未来收益
    X = pd.DataFrame(np.random.randn(100, len(features)), columns=features)
    future_ret = pd.Series(np.random.randn(len(X)))
    proba = pipe.predict_proba(X)[:, 1]
    signal = (proba > threshold).astype(int)
    pnl = (future_ret * signal).cumsum()
    print("示例累计收益:", pnl.iloc[-1])


if __name__ == "__main__":
    main()

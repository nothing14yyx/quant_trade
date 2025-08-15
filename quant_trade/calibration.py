"""概率温度缩放校准模块"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

__all__ = ["TemperatureModel", "fit_temperature", "apply_temperature"]


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 Softmax 实现."""
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        single = True
    else:
        single = False
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs[0] if single else probs


@dataclass
class TemperatureModel:
    """温度缩放模型."""

    temperature: float = 1.0


def fit_temperature(logits: np.ndarray | None, labels: Iterable[int] | None) -> TemperatureModel:
    """通过最小化负对数似然拟合温度参数."""
    if logits is None or labels is None:
        return TemperatureModel()
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return TemperatureModel()
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    y = np.asarray(list(labels), dtype=int)

    def nll(temp_arr: np.ndarray) -> float:
        t = temp_arr[0]
        scaled = logits / t
        probs = _safe_softmax(scaled)
        idx = np.arange(len(y))
        eps = np.finfo(float).eps
        return float(-np.mean(np.log(probs[idx, y] + eps)))

    res = minimize(nll, x0=[1.0], bounds=[(1e-2, 100.0)], method="L-BFGS-B")
    if res.success:
        temp = float(res.x[0])
    else:
        temp = 1.0
    return TemperatureModel(temp)


def apply_temperature(logits: np.ndarray, model: TemperatureModel | None = None) -> np.ndarray:
    """对 logits 应用温度缩放并返回概率."""
    if model is None:
        model = TemperatureModel()
    scaled = np.asarray(logits, dtype=float) / model.temperature
    return _safe_softmax(scaled)

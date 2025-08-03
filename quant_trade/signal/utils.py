import numpy as np
import math
from ..risk_manager import RiskManager

__all__ = [
    "softmax",
    "sigmoid",
    "smooth_score",
    "smooth_series",
    "weighted_quantile",
    "_calc_history_base",
    "risk_budget_threshold",
    "adjust_score",
    "volume_guard",
    "cap_positive",
    "fused_to_risk",
    "sigmoid_dir",
    "sigmoid_confidence",
]

def softmax(x):
    """简单 softmax 实现"""
    arr = np.asarray(x, dtype=float)
    result = np.full_like(arr, np.nan)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return result
    ex = np.exp(valid - np.nanmax(valid))
    result[~np.isnan(arr)] = ex / ex.sum()
    return result


def sigmoid(x):
    """标准 sigmoid 函数"""
    return 1 / (1 + np.exp(-x))


def smooth_score(history_scores, window: int = 3) -> float:
    """简单滑动平均, 返回最近 ``window`` 个得分的均值."""
    if not history_scores:
        return 0.0
    arr = np.asarray(list(history_scores)[-window:], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def smooth_series(history_scores, window: int = 10, alpha: float = 0.2):
    """Return exponentially smoothed series of last ``window`` scores."""
    if not history_scores:
        return []
    arr = np.asarray(list(history_scores)[-window:], dtype=float)
    if arr.size == 0:
        return []
    ema = arr[0]
    smoothed = [ema]
    for val in arr[1:]:
        ema = alpha * val + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def weighted_quantile(values, q, sample_weight=None):
    """Return the weighted quantile of *values* at quantile *q*."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if sample_weight is None:
        return float(np.quantile(values, q))
    sample_weight = np.asarray(sample_weight, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, values))


def _calc_history_base(history, base, quantile, window, decay, limit=None):
    """Helper to compute threshold base from history with optional decay."""
    if history is None:
        return base
    if isinstance(history, np.ndarray):
        arr = history[-window:].astype(float)
    else:
        if len(history) == 0:
            return base
        arr = np.asarray(list(history)[-window:], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return base
    arr = np.abs(arr)
    if arr.size == 0:
        return base
    if decay and decay != 1.0:
        w = np.exp(-decay * np.arange(arr.size)[::-1])
        qv = weighted_quantile(arr, quantile, w)
    else:
        qv = float(np.quantile(arr, quantile))
    if not math.isnan(qv):
        base = max(base, qv)
    if limit is not None and base > limit:
        base = limit
    return base


def risk_budget_threshold(values, *, quantile=0.95, decay=None):
    """根据历史风险指标计算风险预算阈值"""
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.abs(arr)
    if decay:
        w = np.exp(-decay * np.arange(arr.size)[::-1])
        return weighted_quantile(arr, quantile, w)
    return float(np.quantile(arr, quantile))


def cap_positive(
    score: float,
    sentiment: float,
    scale: float = 0.7,
    threshold: float = -0.5,
) -> float:
    """若负面情绪过强则按比例削弱正分"""
    if sentiment <= threshold and score > 0:
        return score * scale
    if sentiment >= 0.5 and score < 0:
        return score * scale
    return score


def adjust_score(
    score: float,
    sentiment: float,
    alpha: float = 0.5,
    *,
    cap_scale: float = 0.7,
    cap_threshold: float = -0.5,
) -> float:
    """根据情绪值调整分数并在负面过强时进一步削弱"""
    if abs(sentiment) <= 0.5:
        return score
    scale = 1 + alpha * np.sign(score) * sentiment
    scale = float(np.clip(scale, 0.6, 1.5))
    adjusted = score * scale
    return cap_positive(adjusted, sentiment, cap_scale, cap_threshold)


def volume_guard(
    score: float,
    ratio: float | None,
    roc: float | None,
    *,
    weak: float = 0.85,
    over: float = 0.9,
    ratio_low: float = -0.5,
    ratio_high: float = 1.5,
    roc_low: float = -20,
    roc_high: float = 100,
) -> float:
    """量能不足或异常时压缩得分"""
    if ratio is None or roc is None:
        return score
    if ratio < ratio_low or roc < roc_low:
        return score * weak
    extreme_ratio = ratio_high * 2
    extreme_roc = roc_high * 1.5
    if ratio_high <= ratio < extreme_ratio and roc_low < roc < extreme_roc:
        mult = 1 + 0.05 * np.sign(score)
        return score * mult
    if ratio >= extreme_ratio or roc >= extreme_roc:
        mult = 1 + (over - 1) * np.sign(score)
        return score * mult
    return score


def fused_to_risk(
    fused_score: float,
    logic_score: float,
    env_score: float,
    *,
    cap: float = 5.0,
) -> float:
    """按安全分母计算并限制 risk score"""
    return RiskManager(cap).fused_to_risk(fused_score, logic_score, env_score)


def sigmoid_dir(score: float, base_th: float, gamma: float) -> float:
    """根据分数计算梯度方向强度, 结果范围 [-1, 1]"""
    amp = np.tanh((abs(score) - base_th) / gamma)
    return np.sign(score) * max(0.0, amp)


def sigmoid_confidence(vote: float, strong_min: float, conf_min: float = 1.0) -> float:
    """根据投票结果计算置信度, 下限由 ``conf_min`` 控制"""
    conf = 1 / (1 + np.exp(-4 * (abs(vote) - strong_min)))
    return max(conf_min, conf)

# -*- coding: utf-8 -*-
"""简单的信号决策模块。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DecisionConfig:
    """决策配置。

    Attributes
    ----------
    upper : float
        触发买入信号的上阈值。
    lower : float
        触发卖出信号的下阈值。
    buy : float
        表示买入的信号数值。
    sell : float
        表示卖出的信号数值。
    hold : float
        表示保持仓位的信号数值。
    """

    upper: float = 0.5
    lower: float = -0.5
    buy: float = 1.0
    sell: float = -1.0
    hold: float = 0.0


def decide_signal(scores: np.ndarray, config: DecisionConfig) -> np.ndarray:
    """根据得分生成交易信号。

    Parameters
    ----------
    scores : np.ndarray
        模型得分或指标值。
    config : DecisionConfig
        决策配置参数。

    Returns
    -------
    np.ndarray
        与 ``scores`` 形状一致的信号数组，取值为 ``config.buy``、
        ``config.sell`` 或 ``config.hold``。
    """

    scores = np.asarray(scores, dtype=float)
    signals = np.full(scores.shape, config.hold, dtype=float)
    signals[scores >= config.upper] = config.buy
    signals[scores <= config.lower] = config.sell
    return signals

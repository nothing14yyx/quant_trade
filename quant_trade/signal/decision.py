# -*- coding: utf-8 -*-
"""信号决策模块。

该文件提供统一的 ``decide_signal`` 函数, 供回测与实盘模块调
用以避免两套分歧的逻辑。函数基于分类概率以及若干预测值
做出买卖/观望的决策, 并返回操作方向、仓位大小以及备注信
息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np


@dataclass
class DecisionConfig:
    """决策参数配置。

    这里的字段与 ``config.yaml`` 中 ``signal`` 节点的键保持一致,
    仅保留在回测与测试中会用到的几个关键参数。
    """

    p_up_min: float = 0.6
    p_down_min: float = 0.6
    margin_min: float = 0.10
    pmin_bump_when_conflict: float = 0.03
    kelly_gamma: float = 0.20
    w_max: float = 0.5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionConfig":
        allowed = {k: data[k] for k in cls.__annotations__.keys() if k in data}
        return cls(**allowed)


def _get_probs(probs: Mapping[str, float] | np.ndarray) -> tuple[float, float]:
    """从数组或映射中提取 ``p_up`` 与 ``p_down``。"""

    if isinstance(probs, Mapping):
        p_up = float(probs.get("up", probs.get("buy", 0.0)))
        p_down = float(probs.get("down", probs.get("sell", 0.0)))
    else:
        arr = np.asarray(probs, dtype=float).reshape(-1)
        if arr.size == 2:  # (p_down, p_up)
            p_down, p_up = float(arr[0]), float(arr[1])
        elif arr.size >= 3:  # (p_down, p_hold, p_up)
            p_down, p_up = float(arr[0]), float(arr[-1])
        else:  # 单个数视为 p_up
            p_up = float(arr[0])
            p_down = 1.0 - p_up
    return p_up, p_down


def decide_signal(
    probs: Mapping[str, float] | np.ndarray,
    rise_pred: float | None,
    drawdown_pred: float | None,
    vol_pred: float | None,
    higher_conflict: bool,
    config: DecisionConfig,
) -> Mapping[str, Any]:
    """根据概率和预测信息生成交易信号。

    Parameters
    ----------
    probs:
        分类概率, 支持 ``{"up": p, "down": q}`` 或 ``[p_down, p_hold, p_up]``
        等形式。
    rise_pred, drawdown_pred, vol_pred:
        模型对未来涨幅、回撤以及波动率的预测, 当前实现仅用于
        占位, 以便后续扩展。
    higher_conflict:
        上级周期是否与当前周期冲突的标志位。
    config:
        决策配置。

    Returns
    -------
    dict
        包含 ``action`` (BUY/SELL/HOLD)、``size`` 以及 ``note`` 的字典。
    """

    p_up, p_down = _get_probs(probs)
    margin = p_up - p_down

    bump = config.pmin_bump_when_conflict if higher_conflict else 0.0
    vol_adj = max(vol_pred or 0.0, 0.0)
    p_up_th = config.p_up_min + bump + vol_adj
    p_down_th = config.p_down_min + bump + vol_adj

    action = "HOLD"
    size = 0.0
    weight = 1.0

    if p_up >= p_up_th and margin >= config.margin_min:
        action = "BUY"
        size = min(config.w_max, config.kelly_gamma * (p_up - p_up_th))
        profit = rise_pred if rise_pred is not None else 0.0
        risk = drawdown_pred if drawdown_pred is not None else 0.0
        if rise_pred is not None or drawdown_pred is not None:
            weight = max(profit - risk, 0.0)
            size *= weight
            size = min(size, config.w_max)
    elif p_down >= p_down_th and -margin >= config.margin_min:
        action = "SELL"
        size = min(config.w_max, config.kelly_gamma * (p_down - p_down_th))
        profit = drawdown_pred if drawdown_pred is not None else 0.0
        risk = rise_pred if rise_pred is not None else 0.0
        if rise_pred is not None or drawdown_pred is not None:
            weight = max(profit - risk, 0.0)
            size *= weight
            size = min(size, config.w_max)

    note_items = []
    if action == "BUY":
        note_items.append(f"p_up={p_up:.2f}")
    elif action == "SELL":
        note_items.append(f"p_down={p_down:.2f}")
    else:
        note_items.append("hold")
    if vol_pred is not None:
        note_items.append(f"vol={vol_pred:.2f}")
    if rise_pred is not None:
        note_items.append(f"rise={rise_pred:.2f}")
    if drawdown_pred is not None:
        note_items.append(f"dd={drawdown_pred:.2f}")
    if action != "HOLD" and (rise_pred is not None or drawdown_pred is not None):
        note_items.append(f"w={weight:.2f}")
    note = ",".join(note_items)

    return {"action": action, "size": size, "note": note}


__all__ = ["DecisionConfig", "decide_signal"]


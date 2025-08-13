from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .utils import sigmoid
from quant_trade.utils import get_cfg_value


def calc_position_size(strength: float, target_risk: float) -> float:
    """根据信号强度计算目标仓位大小并按风险预算压缩。"""
    return min(abs(strength), target_risk)


def compute_exit_multiplier(rsg, vote: float, prev_vote: float, last_signal: int) -> float:
    """根据票数变化决定半退出或全平仓位系数"""
    with rsg._lock:
        exit_lag = rsg._exit_lag

    exit_mult = 1.0
    vote_sign = np.sign(vote)
    prev_sign = np.sign(prev_vote)
    if last_signal == 1:
        if vote_sign == 1 and prev_vote > vote:
            exit_mult = 0.5
            exit_lag = 0
        elif vote_sign <= 0 and prev_sign > 0:
            exit_lag += 1
            exit_mult = 0.0 if exit_lag >= rsg.exit_lag_bars else 0.5
        else:
            exit_lag = 0
    elif last_signal == -1:
        if vote_sign == -1 and prev_vote < vote:
            exit_mult = 0.5
            exit_lag = 0
        elif vote_sign >= 0 and prev_sign < 0:
            exit_lag += 1
            exit_mult = 0.0 if exit_lag >= rsg.exit_lag_bars else 0.5
        else:
            exit_lag = 0
    else:
        exit_lag = 0

    with rsg._lock:
        rsg._exit_lag = exit_lag

    return exit_mult


def _apply_risk_adjustment(pos_size: float, risk_score: float, risk_scale: float) -> float:
    """根据风险评分调整仓位大小"""
    risk_factor = math.exp(-risk_scale * risk_score)
    return pos_size * risk_factor


def _apply_low_volume_penalty(
    rsg,
    pos_size: float,
    *,
    regime: str,
    vol_ratio: float | None,
    fused_score: float,
    base_th: float,
    consensus_all: bool,
) -> Tuple[float, bool]:
    """在低成交量环境下惩罚仓位, 返回是否触发标记"""
    low_vol_flag = (
        regime == "range"
        and vol_ratio is not None
        and vol_ratio < rsg.low_vol_ratio
        and abs(fused_score) < base_th + 0.02
        and not consensus_all
    )
    if low_vol_flag:
        pos_size *= 0.5
    return pos_size, low_vol_flag


def _apply_vol_prediction_adjustment(pos_size: float, vol_p: float | None) -> float:
    """根据预测波动率对仓位进行修正"""
    if vol_p is not None:
        pos_size *= max(0.4, 1 - min(0.6, vol_p))
    return pos_size


def _adjust_min_pos_vol(
    min_pos: float,
    atr: float | None,
    vol_p: float | None,
    *,
    min_pos_vol_scale: float,
) -> float:
    """根据历史 ATR 或预测波动率调节仓位下限"""
    vol_ref = 0.0
    if vol_p is not None and np.isfinite(vol_p):
        vol_ref = abs(vol_p)
    elif atr is not None and np.isfinite(atr):
        vol_ref = abs(atr)
    return min_pos * (1 + min_pos_vol_scale * vol_ref)


def compute_tp_sl(
    rsg,
    price,
    atr,
    direction,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    *,
    rise_pred: float | None = None,
    drawdown_pred: float | None = None,
    regime: str | None = None,
):
    """计算止盈止损价格，可根据模型预测值微调"""
    if direction == 0:
        return None, None
    if price is None or not np.isfinite(price):
        return None, None
    if price <= 0:
        return None, None
    if atr is None or not np.isfinite(atr):
        return None, None
    if atr == 0:
        atr = 0.005 * price

    cfg = getattr(rsg, "cfg", {})
    max_sl_pct = get_cfg_value(cfg, "max_stop_loss_pct", 0.05)

    if rise_pred is None:
        rise_pred = 0.0
    if drawdown_pred is None:
        drawdown_pred = 0.0

    if direction == 1:
        max_sl = price * (1 - max_sl_pct)
        tp0 = price * (1 + min(rise_pred, 0.10))
        sl0 = price * (1 + max(drawdown_pred, -max_sl_pct))
        tp = max(tp0, price * 1.02)
        sl = min(sl0, max_sl)
    else:
        max_sl = price * (1 + max_sl_pct)
        tp0 = price * (1 - min(rise_pred, 0.10))
        sl0 = price * (1 - max(drawdown_pred, -max_sl_pct))
        tp = min(tp0, price * 0.98)
        sl = max(sl0, max_sl)

    return float(tp), float(sl)

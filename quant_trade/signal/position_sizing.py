from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .utils import sigmoid_dir
from quant_trade.utils import get_cfg_value


def calc_position_size(
    fused_score: float,
    base_th: float,
    *,
    max_position: float,
    gamma: float = 0.05,
    cvar_target: float | None = None,
    vol_target: float | None = None,
    min_exposure: float | None = None,
) -> float:
    """根据信号强度及风险目标计算最终仓位。

    参数
    ----
    fused_score: 综合后的信号分数。
    base_th: 基准阈值，用于确定信号有效性。
    max_position: 仓位上限。
    gamma: ``sigmoid_dir`` 平滑系数。
    cvar_target: 来自 CVaR 管控的目标仓位(可选)。
    vol_target: 来自波动率管控的目标仓位(可选)。
    min_exposure: 信号弱但方向未被否定时的最低敞口(可选)，通常由配置
        中的 ``min_exposure_base`` 与实时波动率计算得出。
    """

    strength = sigmoid_dir(fused_score, base_th, gamma)
    target_risk = abs(strength) * max_position
    final_size = target_risk

    if cvar_target is not None:
        final_size = min(final_size, cvar_target)
    if vol_target is not None:
        final_size = min(final_size, vol_target)

    if min_exposure is not None and strength != 0:
        final_size = max(final_size, min(min_exposure, max_position))

    return final_size


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


def apply_normalized_multipliers(pos_size: float, factors: dict) -> Tuple[float, dict]:
    """应用并规范化多个倍率因子。

    参数
    ----
    pos_size: 初始仓位大小。
    factors: 包含倍率计算所需参数的字典，可包含以下键：
        - ``low_volume``: dict，低成交量惩罚相关参数；
        - ``vol_prediction``: float，可选，预测波动率；
        - ``extra``: Iterable[float]，额外需直接应用的倍率。

    返回
    ----
    调整后的仓位以及标记字典（目前仅包含 ``low_volume``）。
    """

    flags: dict = {}
    mults: list[float] = []

    # 低成交量惩罚逻辑
    if "low_volume" in factors:
        lv = factors["low_volume"]
        rsg = lv.get("rsg")
        regime = lv.get("regime")
        vol_ratio = lv.get("vol_ratio")
        fused_score = lv.get("fused_score")
        base_th = lv.get("base_th")
        consensus_all = lv.get("consensus_all", False)
        low_vol_flag = (
            regime == "range"
            and vol_ratio is not None
            and vol_ratio < getattr(rsg, "low_vol_ratio", 0)
            and abs(fused_score) < base_th + 0.02
            and not consensus_all
        )
        flags["low_volume"] = low_vol_flag
        if low_vol_flag:
            mults.append(0.5)

    # 预测波动率修正逻辑
    if "vol_prediction" in factors:
        vol_p = factors["vol_prediction"]
        if vol_p is not None:
            mults.append(max(0.4, 1 - min(0.6, vol_p)))

    # 其他额外倍率
    for m in factors.get("extra", []):
        if m is not None and np.isfinite(m):
            mults.append(m)

    norm_mults = [max(0.0, min(1.0, m)) for m in mults]
    final_mult = float(np.prod(norm_mults)) if norm_mults else 1.0

    return pos_size * final_mult, flags


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

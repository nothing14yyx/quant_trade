from __future__ import annotations

import math
from typing import Tuple, List

import numpy as np

from ..constants import RiskReason
from .utils import sigmoid
from quant_trade.utils import get_cfg_value


class PositionSizerImpl:
    """仓位管理相关逻辑实现"""

    def __init__(self, rsg):
        self.rsg = rsg

    def compute_exit_multiplier(self, vote: float, prev_vote: float, last_signal: int) -> float:
        """根据票数变化决定半退出或全平仓位系数"""
        with self.rsg._lock:
            exit_lag = self.rsg._exit_lag

        exit_mult = 1.0
        vote_sign = np.sign(vote)
        prev_sign = np.sign(prev_vote)
        if last_signal == 1:
            if vote_sign == 1 and prev_vote > vote:
                exit_mult = 0.5
                exit_lag = 0
            elif vote_sign <= 0 and prev_sign > 0:
                exit_lag += 1
                exit_mult = 0.0 if exit_lag >= self.rsg.exit_lag_bars else 0.5
            else:
                exit_lag = 0
        elif last_signal == -1:
            if vote_sign == -1 and prev_vote < vote:
                exit_mult = 0.5
                exit_lag = 0
            elif vote_sign >= 0 and prev_sign < 0:
                exit_lag += 1
                exit_mult = 0.0 if exit_lag >= self.rsg.exit_lag_bars else 0.5
            else:
                exit_lag = 0
        else:
            exit_lag = 0

        with self.rsg._lock:
            self.rsg._exit_lag = exit_lag

        return exit_mult

    def _apply_risk_adjustment(self, pos_size: float, risk_score: float) -> float:
        """根据风险评分调整仓位大小"""
        risk_factor = math.exp(-self.rsg.risk_scale * risk_score)
        return pos_size * risk_factor

    def _apply_low_volume_penalty(
        self,
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
            and vol_ratio < self.rsg.low_vol_ratio
            and abs(fused_score) < base_th + 0.02
            and not consensus_all
        )
        if low_vol_flag:
            pos_size *= 0.5
        return pos_size, low_vol_flag

    def _apply_vol_prediction_adjustment(self, pos_size: float, vol_p: float | None) -> float:
        """根据预测波动率对仓位进行修正"""
        if vol_p is not None:
            pos_size *= max(0.4, 1 - min(0.6, vol_p))
        return pos_size

    def _adjust_min_pos_vol(self, min_pos: float, atr: float | None, vol_p: float | None) -> float:
        """根据历史 ATR 或预测波动率调节仓位下限"""
        vol_ref = 0.0
        if vol_p is not None and np.isfinite(vol_p):
            vol_ref = abs(vol_p)
        elif atr is not None and np.isfinite(atr):
            vol_ref = abs(atr)
        return min_pos * (1 + self.rsg.min_pos_vol_scale * vol_ref)

    def compute_tp_sl(
        self,
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

        cfg = getattr(self.rsg, "cfg", {})
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

    def _determine_direction(
        self,
        grad_dir: float,
        regime: str,
        fs: dict,
        st_dir: int,
        vol_breakout_val: float | None,
        conf_vote: float,
        weak_vote: bool,
        fused_score: float,
        base_th: float,
        raw_f1h: dict | None,
        std_1h: dict,
        ts,
        symbol,
    ) -> int:
        """Determine final trade direction."""

        if not self.rsg.direction_filters_enabled:
            return 0 if grad_dir == 0 else int(np.sign(grad_dir))

        direction = 0 if grad_dir == 0 else int(np.sign(grad_dir))
        if weak_vote:
            direction = 0

        if regime == "range":
            atr_v = (raw_f1h or std_1h).get("atr_pct_1h")
            bb_w = (raw_f1h or std_1h).get("bb_width_1h")
            low_vol = False
            if atr_v is not None and atr_v < 0.005:
                low_vol = True
            if bb_w is not None and bb_w < 0.01:
                low_vol = True
            if low_vol:
                direction = 0
            elif vol_breakout_val is None or vol_breakout_val <= 0 or conf_vote < 0.15:
                direction = 0

        if self.rsg._cooldown > 0:
            self.rsg._cooldown -= 1

        if self.rsg._last_signal != 0 and direction != 0 and direction != self.rsg._last_signal:
            flip_th = max(base_th, self.rsg.flip_coeff * abs(self.rsg._last_score))
            if abs(fused_score) < flip_th or self.rsg._cooldown > 0:
                direction = self.rsg._last_signal
            else:
                self.rsg._cooldown = 2

        align_count = 0
        if direction != 0:
            for p in ("1h", "4h", "d1"):
                if np.sign(fs[p]["trend"]) == direction:
                    align_count += 1
            if st_dir != 0 and st_dir == direction:
                align_count += 1
            min_align = self.rsg.min_trend_align if regime == "trend" else max(
                self.rsg.min_trend_align - 1, 0
            )
            if align_count < min_align:
                direction = 0

        return direction

    def _apply_position_filters(
        self,
        pos_size: float,
        direction: int,
        *,
        weak_vote: bool,
        funding_conflicts: int,
        oi_overheat: bool,
        risk_score: float,
        logic_score: float,
        base_th: float,
        conflict_filter_triggered: bool,
        zero_reason: str | None,
    ) -> Tuple[float, int, str | None, List[str]]:
        """Apply filters to position size and direction."""

        penalties: List[str] = []

        if not self.rsg.direction_filters_enabled:
            return pos_size, direction, zero_reason, penalties

        if weak_vote:
            if self.rsg.filter_penalty_mode:
                pos_size *= self.rsg.penalty_factor
                penalties.append(RiskReason.VOTE_PENALTY.value)
                zero_reason = None
            else:
                direction = 0
                pos_size = 0.0
                zero_reason = zero_reason or RiskReason.VOTE_FILTER.value

        if funding_conflicts > self.rsg.veto_level:
            if self.rsg.filter_penalty_mode:
                pos_size *= self.rsg.penalty_factor
                penalties.append(RiskReason.FUNDING_PENALTY.value)
                zero_reason = None
            else:
                direction = 0
                pos_size = 0.0
                zero_reason = zero_reason or RiskReason.FUNDING_CONFLICT.value

        if oi_overheat:
            pos_size *= 0.5
            penalties.append(RiskReason.OI_OVERHEAT.value)

        pos_map = base_th * 2.0
        if risk_score > 1 or logic_score < -0.3:
            pos_map = min(pos_map, 0.5)
        pos_size = min(pos_size, pos_map)

        if conflict_filter_triggered:
            if self.rsg.filter_penalty_mode:
                pos_size *= self.rsg.penalty_factor
                penalties.append(RiskReason.CONFLICT_PENALTY.value)
                zero_reason = None
            else:
                pos_size = 0.0
                direction = 0
                zero_reason = zero_reason or RiskReason.CONFLICT_FILTER.value

        return pos_size, direction, zero_reason, penalties

    def decide(
        self,
        *,
        grad_dir: float,
        base_coeff: float,
        confidence_factor: float,
        vol_ratio: float | None,
        fused_score: float,
        base_th: float,
        regime: str,
        vol_p: float | None,
        atr: float | None,
        risk_score: float,
        crowding_factor: float,
        cfg_th_sig: dict,
        direction: int,
        exit_mult: float,
        consensus_all: bool = False,
    ) -> Tuple[float, int, float, str | None]:
        """Calculate final position size and tier."""

        tier = base_coeff * abs(grad_dir)
        base_size = tier

        zero_reason: str | None = None
        low_vol_flag = False

        pos_size = base_size * sigmoid(confidence_factor)
        pos_size = self._apply_risk_adjustment(pos_size, risk_score)
        pos_size *= exit_mult
        pos_size = min(pos_size, self.rsg.max_position)
        pos_size *= crowding_factor
        if direction == 0:
            pos_size = 0.0
            zero_reason = RiskReason.NO_DIRECTION.value

        pos_size, low_vol_flag = self._apply_low_volume_penalty(
            pos_size,
            regime=regime,
            vol_ratio=vol_ratio,
            fused_score=fused_score,
            base_th=base_th,
            consensus_all=consensus_all,
        )

        pos_size = self._apply_vol_prediction_adjustment(pos_size, vol_p)

        min_pos = cfg_th_sig.get("min_pos", self.rsg.signal_params.min_pos)
        min_pos = self._adjust_min_pos_vol(min_pos, atr, vol_p)
        dynamic_min = min_pos * math.exp(self.rsg.risk_scale * risk_score)
        if self.rsg.risk_filters_enabled and pos_size < dynamic_min:
            pos_size = min(max(pos_size, dynamic_min), self.rsg.max_position)
            zero_reason = RiskReason.MIN_POS.value

        return pos_size, direction, tier, zero_reason

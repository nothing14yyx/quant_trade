from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskManager:
    """风险控制与仓位计算逻辑"""

    risk_score_cap: float = 5.0
    exit_lag_bars: int = 0
    oi_scale: float = 0.8
    max_position: float = 0.3

    def compute_tp_sl(self, price, atr, direction, tp_mult=1.5, sl_mult=1.0, *, rise_pred=None, drawdown_pred=None):
        if direction == 0 or price is None or price <= 0:
            return None, None
        if atr is None or atr == 0:
            atr = 0.005 * price
        tp_mult = float(np.clip(tp_mult, 0.5, 3.0))
        sl_mult = float(np.clip(sl_mult, 0.5, 2.0))
        if rise_pred is not None and drawdown_pred is not None:
            if direction == 1:
                take_profit = price * (1 + max(rise_pred, 0))
                stop_loss = price * (1 + min(drawdown_pred, 0))
            else:
                take_profit = price * (1 - max(drawdown_pred, 0))
                stop_loss = price * (1 - min(rise_pred, 0))
            if abs(take_profit - price) < 1e-8 and abs(stop_loss - price) < 1e-8:
                if direction == 1:
                    take_profit = price + tp_mult * atr
                    stop_loss = price - sl_mult * atr
                else:
                    take_profit = price - tp_mult * atr
                    stop_loss = price + sl_mult * atr
        else:
            if direction == 1:
                take_profit = price + tp_mult * atr
                stop_loss = price - sl_mult * atr
            else:
                take_profit = price - tp_mult * atr
                stop_loss = price + sl_mult * atr
        return float(take_profit), float(stop_loss)

    def apply_oi_overheat_protection(self, fused_score, oi_chg, th_oi):
        if th_oi is None or abs(oi_chg) < th_oi:
            return fused_score * (1 + 0.03 * oi_chg), False
        return fused_score * self.oi_scale, True

    @staticmethod
    def crowding_protection(scores, current_score, base_th=0.2, max_rate=0.95):
        if not scores or len(scores) < 30:
            return 1.0
        arr = np.array(scores, dtype=float)
        mask = np.abs(arr) >= base_th * 0.8
        arr = arr[mask]
        signs = [s for s in np.sign(arr) if s != 0]
        total = len(signs)
        if total == 0:
            return 1.0
        pos_counts = Counter(signs)
        dominant_dir, cnt = pos_counts.most_common(1)[0]
        if np.sign(current_score) != dominant_dir:
            return 1.0
        ratio = cnt / total
        rank_pct = pd.Series(np.abs(list(arr) + [current_score])).rank(pct=True).iloc[-1]
        ratio_intensity = max(0.0, (ratio - max_rate) / (1 - max_rate))
        rank_intensity = max(0.0, rank_pct - 0.8) / 0.2
        intensity = min(1.0, max(ratio_intensity, rank_intensity))
        factor = 1.0 - 0.2 * intensity
        return factor

    @staticmethod
    def dynamic_threshold(atr, adx, funding=0, *, atr_4h=None, adx_4h=None, atr_d1=None, adx_d1=None, pred_vol=None, pred_vol_4h=None, pred_vol_d1=None, vix_proxy=None, base=0.08, regime=None, low_base=0.06, reversal=False, history_scores=None, params=None):
        if params is None:
            params = {
                "base_th": 0.08,
                "gamma": 0.05,
                "quantile": 0.80,
                "low_base": low_base,
                "rev_boost": 0.30,
                "rev_th_mult": 0.60,
                "atr_mult": 4.0,
                "funding_mult": 8.0,
                "adx_div": 100.0,
            }
        if base is None:
            base = params.get("base_th", 0.08)
        if low_base is None:
            low_base = params.get("low_base", 0.06)
        hist_base = base
        if history_scores:
            arr = np.asarray(list(history_scores)[-60:], dtype=float)
            arr = np.abs(arr)
            if arr.size > 0:
                qv = float(np.quantile(arr, params.get("quantile", 0.8)))
                if not math.isnan(qv):
                    hist_base = max(base, qv)
        if hist_base > 0.12:
            hist_base = 0.12
        th = hist_base
        atr_eff = abs(atr)
        if atr_4h is not None:
            atr_eff += 0.5 * abs(atr_4h)
        if atr_d1 is not None:
            atr_eff += 0.25 * abs(atr_d1)
        th += min(0.10, atr_eff * params.get("atr_mult", 4.0))
        fund_eff = abs(funding)
        if pred_vol is not None:
            fund_eff += 0.5 * abs(pred_vol)
        if pred_vol_4h is not None:
            fund_eff += 0.25 * abs(pred_vol_4h)
        if pred_vol_d1 is not None:
            fund_eff += 0.15 * abs(pred_vol_d1)
        if vix_proxy is not None:
            fund_eff += 0.25 * abs(vix_proxy)
        th += min(0.08, fund_eff * params.get("funding_mult", 8.0))
        adx_eff = abs(adx)
        if adx_4h is not None:
            adx_eff += 0.5 * abs(adx_4h)
        if adx_d1 is not None:
            adx_eff += 0.25 * abs(adx_d1)
        th += min(0.04, adx_eff / params.get("adx_div", 100.0))
        if atr_eff == 0 and adx_eff == 0 and fund_eff == 0:
            th = min(th, hist_base)
        if reversal:
            th *= params.get("rev_th_mult", 0.6)
        rev_boost = params.get("rev_boost", 0.30)
        if regime == "trend":
            th *= 1.05
            rev_boost *= 0.8
        elif regime == "range":
            th *= 0.95
            rev_boost *= 1.2
        return max(th, low_base), rev_boost

    @staticmethod
    def compute_position_size(*, grad_dir, base_coeff, confidence_factor, vol_ratio, fused_score, base_th, regime, oi_overheat, vol_p, risk_score, crowding_factor, cfg_th_sig, scores, direction, exit_mult, max_position=0.3, consensus_all=False):
        tier = base_coeff * abs(grad_dir)
        base_size = tier
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        pos_size = base_size * _sigmoid(confidence_factor) * (1.0 / (1.0 + risk_score))
        pos_size *= exit_mult
        pos_size = min(pos_size, max_position)
        pos_size *= crowding_factor
        if direction == 0:
            pos_size = 0.0
        if regime == "range" and vol_ratio is not None and vol_ratio < 0.2 and abs(fused_score) < base_th + 0.02 and not consensus_all:
            pos_size *= 0.5
        if vol_p is not None:
            pos_size *= max(0.4, 1 - min(0.6, vol_p))
        min_pos = cfg_th_sig.get("min_pos", 0.05)
        if pos_size < min_pos:
            direction, pos_size = 0, 0.0
        return pos_size, direction, tier, None

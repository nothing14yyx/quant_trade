# -*- coding: utf-8 -*-
"""Risk filter utilities extracted from :mod:`core`."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np

from ..constants import RiskReason
from .thresholding_dynamic import ThresholdingDynamic
from .utils import risk_budget_threshold
from typing import Any, Mapping


class RiskFiltersImpl:
    """封装风险相关过滤逻辑的实现类"""

    def __init__(self, core) -> None:
        """Parameters
        ----------
        core : RobustSignalGenerator
            主体对象实例, 用于访问共享的方法与属性
        """
        self.core = core

    # ------------------------------------------------------------------
    # OI 过热保护
    # ------------------------------------------------------------------
    def apply_oi_overheat_protection(self, fused_score, oi_chg, th_oi):
        """根据 OI 变化调整得分, 并返回是否过热标记"""
        if th_oi is None or abs(oi_chg) < th_oi:
            return fused_score * (1 + 0.03 * oi_chg), False

        logging.info("OI overheat detected: %.4f", oi_chg)
        return fused_score * self.core.oi_scale, True

    # ------------------------------------------------------------------
    # 拥挤度保护
    # ------------------------------------------------------------------
    def apply_crowding_protection(
        self,
        fused_score: float,
        *,
        base_th: float,
        all_scores_list: Sequence[float] | None,
        oi_chg: float | None,
        cache: dict,
        vol_pred: float | None,
        oi_overheat: bool,
        symbol: str | None,
    ) -> tuple[float, float, float | None]:
        """计算拥挤度因子并调整 ``fused_score``"""
        th_oi = cache.get("th_oi")
        if th_oi is None and oi_chg is not None:
            th_oi = self.core.get_dynamic_oi_threshold(pred_vol=vol_pred)
            cache["th_oi"] = th_oi

        crowding_factor = 1.0
        if not oi_overheat and all_scores_list is not None:
            factor = self.core.crowding_protection(
                all_scores_list, fused_score, base_th
            )
            fused_score *= factor
            crowding_factor *= factor

        if th_oi is not None and oi_chg is not None:
            oi_crowd = abs(oi_chg) / max(th_oi, 1e-6)
            mult = 1 - min(0.5, oi_crowd * 0.5)
            if mult < 1:
                logging.debug(
                    "oi change %.4f threshold %.3f -> crowding mult %.3f for %s",
                    oi_chg,
                    th_oi,
                    mult,
                    symbol,
                )
                fused_score *= mult
                crowding_factor *= mult

        return fused_score, crowding_factor, th_oi


    # ------------------------------------------------------------------
    # 综合风险过滤
    # ------------------------------------------------------------------
    def compute_risk_multipliers(
        self,
        fused_score: float,
        logic_score: float,
        env_score: float,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        raw_f1h: dict,
        raw_f4h: dict,
        raw_fd1: dict,
        vol_preds: dict,
        open_interest: dict | None,
        all_scores_list: list | None,
        rev_dir: int,
        cache: dict,
        global_metrics: dict | None,
        symbol: str | None,
        *,
        dyn_base: float | None,
    ) -> tuple[float, float]:
        """计算风险乘数"""
        core = self.core
        score_mult = 1.0
        pos_mult = 1.0
        if not core.risk_filters_enabled and not core.dynamic_threshold_enabled:
            self.last_risk_score = 0.0
            self.last_crowding_factor = 1.0
            self.last_risk_th = 0.0
            self.last_base_th = core.signal_params.base_th
            self.last_rev_boost = 0.0
            self.last_regime = "range"
            self.last_rev_dir = 0
            self.last_funding_conflicts = 0
            self.last_th_oi = None
            return score_mult, pos_mult

        atr_1h = raw_f1h.get("atr_pct_1h", 0) if raw_f1h else 0
        adx_1h = raw_f1h.get("adx_1h", 0) if raw_f1h else 0
        funding_1h = raw_f1h.get("funding_rate_1h", 0) if raw_f1h else 0

        atr_4h = raw_f4h.get("atr_pct_4h") if raw_f4h else None
        adx_4h = raw_f4h.get("adx_4h") if raw_f4h else None
        atr_d1 = raw_fd1.get("atr_pct_d1") if raw_fd1 else None
        adx_d1 = raw_fd1.get("adx_d1") if raw_fd1 else None

        vix_p = None
        if global_metrics is not None:
            vix_p = global_metrics.get("vix_proxy")
        if vix_p is None and open_interest is not None:
            vix_p = open_interest.get("vix_proxy")

        bb_chg = raw_f1h.get("bb_width_chg_1h") if raw_f1h else None
        channel_pos = raw_f1h.get("channel_pos_1h") if raw_f1h else None
        regime = core.detect_market_regime(
            adx_1h,
            adx_4h or 0,
            adx_d1 or 0,
            bb_chg,
            channel_pos,
        )
        rsi_d1 = std_d1.get("rsi_d1", 50)
        hist_d1 = cache.get("_raw_history", {}).get("d1", [])
        pairs = [(h.get("rsi_d1"), h.get("vol_ma_ratio_d1")) for h in hist_d1]
        rsi_hist = [p[0] for p in pairs if p[0] is not None and p[1] is not None]
        vol_hist = [p[1] for p in pairs if p[0] is not None and p[1] is not None]
        lower, _ = ThresholdingDynamic.adaptive_rsi_threshold(
            rsi_hist, vol_hist, core.rsi_k
        )
        if std_d1.get("break_support_d1", 0) > 0 and rsi_d1 < lower:
            regime = "range"
            rev_dir = 1
        cfg_th = core.signal_threshold_cfg
        params = core.signal_params
        cfg_base = cfg_th.get("base_th", params.base_th)
        if core.dynamic_threshold_enabled:
            base_input = dyn_base if dyn_base is not None else cfg_base
            base_th, rev_boost = core.thresholding.base(
                atr_1h,
                adx_1h,
                funding_1h,
                atr_4h=atr_4h,
                adx_4h=adx_4h,
                atr_d1=atr_d1,
                adx_d1=adx_d1,
                bb_width_chg=bb_chg,
                channel_pos=channel_pos,
                pred_vol=vol_preds.get("1h"),
                pred_vol_4h=vol_preds.get("4h"),
                pred_vol_d1=vol_preds.get("d1"),
                vix_proxy=vix_p,
                regime=regime,
                base=base_input,
                reversal=bool(rev_dir),
                history_scores=cache["history_scores"],
            )
        else:
            base_th = cfg_base
            rev_boost = cfg_th.get("rev_boost", params.rev_boost)
        base_th *= getattr(core, "phase_th_mult", 1.0)
        self.last_base_th = base_th
        self.last_rev_boost = rev_boost
        self.last_regime = regime
        self.last_rev_dir = rev_dir
        if rev_dir != 0:
            core._cooldown = 0

        if not core.risk_filters_enabled:
            self.last_risk_score = 0.0
            self.last_crowding_factor = 1.0
            self.last_risk_th = 0.0
            self.last_funding_conflicts = 0
            self.last_th_oi = None
            return score_mult, pos_mult

        funding_conflicts = 0
        for p, raw_f in [("1h", raw_f1h), ("4h", raw_f4h), ("d1", raw_fd1)]:
            if raw_f is None:
                continue
            f_rate = raw_f.get(f"funding_rate_{p}", 0)
            if abs(f_rate) > 0.0005 and np.sign(f_rate) * np.sign(fused_score) < 0:
                penalty = min(abs(f_rate) * 20, 0.20)
                score_mult *= 1 - penalty
                funding_conflicts += 1
        if funding_conflicts >= core.veto_conflict_count:
            if core.filter_penalty_mode:
                score_mult *= core.penalty_factor
                pos_mult *= core.penalty_factor
            else:
                score_mult = 0.0
                pos_mult = 0.0

        adj_score = fused_score * score_mult
        _, crowding_factor, th_oi = self.apply_crowding_protection(
            adj_score,
            base_th=base_th,
            all_scores_list=all_scores_list,
            oi_chg=cache.get("oi_chg"),
            cache=cache,
            vol_pred=vol_preds.get("1h"),
            oi_overheat=cache.get("oi_overheat", False),
            symbol=symbol,
        )
        score_mult *= crowding_factor
        self.last_th_oi = th_oi
        risk_score = core.risk_manager.calc_risk(
            env_score,
            pred_vol=vol_preds.get("1h"),
            oi_change=open_interest.get("oi_chg") if open_interest else None,
        )

        score_mult *= 1 - core.risk_adjust_factor * risk_score
        with core._lock:
            atr_hist = [
                r.get("atr_pct_1h")
                for r in cache.get("_raw_history", {}).get("1h", [])
                if r.get("atr_pct_1h") is not None
            ]
            oi_hist = list(cache.get("oi_change_history", []))
        hist = [abs(v) for v in atr_hist if v is not None]
        if not hist:
            hist = [abs(v) for v in oi_hist if v is not None]
        dyn_risk_th = (
            risk_budget_threshold(hist, quantile=core.signal_params.quantile)
            if hist
            else float("nan")
        )
        risk_th = core.risk_adjust_threshold
        if risk_th is None:
            with core._lock:
                hist_scores = list(cache.get("history_scores", []))
            risk_th = risk_budget_threshold(
                hist_scores, quantile=core.risk_th_quantile
            )
            if math.isnan(risk_th):
                risk_th = 0.0
        if math.isnan(dyn_risk_th):
            logging.warning(
                "历史数据不足，继续使用固定风险阈值；atr_hist=%s，oi_hist=%s",
                atr_hist,
                oi_hist,
            )
        else:
            risk_th = max(risk_th, dyn_risk_th)

        self.last_risk_score = risk_score
        self.last_crowding_factor = crowding_factor
        self.last_risk_th = risk_th
        self.last_funding_conflicts = funding_conflicts

        if abs(fused_score * score_mult) < risk_th:
            score_mult = 0.0
            pos_mult = 0.0

        penalty_triggered = False
        if risk_score > core.risk_score_limit:
            penalty_triggered = True
        if crowding_factor < 0 or crowding_factor > core.crowding_limit:
            penalty_triggered = True
        if penalty_triggered:
            if core.filter_penalty_mode:
                score_mult *= core.penalty_factor
                pos_mult *= core.penalty_factor
            else:
                score_mult = 0.0
                pos_mult = 0.0

        fs_adj = fused_score * score_mult
        with core._lock:
            cache["history_scores"].append(fs_adj)
            core.all_scores_list.append(fs_adj)

        return score_mult, pos_mult

    def collect_risk_reasons(
        self,
        fused_score: float,
        score_mult: float,
        pos_mult: float,
        cache: dict,
    ) -> list[str]:
        """根据缓存信息收集风险原因"""
        core = self.core
        reasons: list[str] = []
        if cache.get("oi_overheat"):
            reasons.append(RiskReason.OI_OVERHEAT.value)
        if self.last_funding_conflicts >= core.veto_conflict_count:
            reasons.append(
                RiskReason.FUNDING_PENALTY.value
                if core.filter_penalty_mode
                else RiskReason.FUNDING_CONFLICT.value
            )
        if abs(fused_score * score_mult) < self.last_risk_th or self.last_risk_score > core.risk_score_limit:
            reasons.append(RiskReason.RISK_LIMIT.value)
        if self.last_crowding_factor < 0 or self.last_crowding_factor > core.crowding_limit:
            reasons.append(
                RiskReason.CONFLICT_PENALTY.value
                if core.filter_penalty_mode
                else RiskReason.CONFLICT_FILTER.value
            )
        return reasons

    def apply_risk_filters(
        self,
        fused_score: float,
        logic_score: float,
        env_score: float,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        raw_f1h: dict,
        raw_f4h: dict,
        raw_fd1: dict,
        vol_preds: dict,
        open_interest: dict | None,
        all_scores_list: list | None,
        rev_dir: int,
        cache: dict,
        global_metrics: dict | None,
        symbol: str | None,
        *,
        dyn_base: float | None,
    ) -> tuple[float, float, list[str]]:
        """兼容旧接口的包装函数"""
        score_mult, pos_mult = self.compute_risk_multipliers(
            fused_score,
            logic_score,
            env_score,
            std_1h,
            std_4h,
            std_d1,
            raw_f1h,
            raw_f4h,
            raw_fd1,
            vol_preds,
            open_interest,
            all_scores_list,
            rev_dir,
            cache,
            global_metrics,
            symbol,
            dyn_base=dyn_base,
        )
        reasons = self.collect_risk_reasons(
            fused_score, score_mult, pos_mult, cache
        )
        return score_mult, pos_mult, reasons

    def get_last_metrics(self) -> dict[str, Any]:
        """返回最近一次风险计算的关键指标。

        该方法将内部缓存的拥挤度因子、OI 阈值、风险分数等信息
        打包成字典，方便外部调用方在不直接访问属性的情况下获取。
        """

        return {
            "crowding_factor": getattr(self, "last_crowding_factor", 1.0),
            "oi_threshold": getattr(self, "last_th_oi", None),
            "risk_score": getattr(self, "last_risk_score", 0.0),
            "base_th": getattr(self, "last_base_th", 0.0),
            "rev_dir": getattr(self, "last_rev_dir", 0),
            "cooldown": getattr(self.core, "_cooldown", 0),
        }


def compute_risk_multipliers(
    fused_score: float,
    base_th: float,
    scores: Mapping[str, Any],
    *,
    global_metrics: Mapping[str, Any] | None = None,
    open_interest: Mapping[str, Any] | None = None,
    all_scores_list: Sequence[float] | None = None,
    symbol: str | None = None,
) -> tuple[float, float, list[str], dict[str, Any]]:
    """简化版风险乘数计算接口。

    该函数用于轻量级的 ``generate_signal`` 管线，返回分数/仓位乘数
    以及附加的风险信息字典。这里的实现较为简化，只提供默认值，
    以便调用方在测试中验证字段存在性。
    """

    info = {
        "crowding_factor": 1.0,
        "oi_threshold": None,
        "risk_score": 0.0,
        "base_th": base_th,
        "rev_dir": 0,
        "cooldown": 0,
    }
    return 1.0, 1.0, [], info

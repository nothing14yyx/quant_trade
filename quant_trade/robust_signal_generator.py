"""Backward-compatible wrappers for the new signal pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import math
import threading
import types

from .signal import core, calc_position_size
from .signal.dynamic_thresholds import DynamicThresholdParams, SignalThresholdParams
from .signal.position_sizing import apply_normalized_multipliers
from .constants import RiskReason
from .risk_manager import cvar_limit
from .signal.voting_model import safe_load, DEFAULT_MODEL_PATH


@dataclass
class RobustSignalGeneratorConfig:
    """Minimal configuration placeholder for tests."""

    model_paths: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    feature_cols_1h: list[str] = field(default_factory=list)
    feature_cols_4h: list[str] = field(default_factory=list)
    feature_cols_d1: list[str] = field(default_factory=list)
    prob_margin: float = 0.1
    strong_prob_th: float = 0.8
    config_path: str | Path | None = None

    @classmethod
    def from_cfg(
        cls,
        cfg: Mapping[str, Any],
        cfg_path: str | Path | None = None,
    ) -> "RobustSignalGeneratorConfig":
        return cls(
            model_paths=cfg.get("model_paths", {}),
            feature_cols_1h=cfg.get("feature_cols_1h", []),
            feature_cols_4h=cfg.get("feature_cols_4h", []),
            feature_cols_d1=cfg.get("feature_cols_d1", []),
            prob_margin=cfg.get("prob_margin", 0.1),
            strong_prob_th=cfg.get("strong_prob_th", 0.8),
            config_path=cfg_path,
        )


class RobustSignalGenerator:
    """Thin wrapper around the functional signal pipeline."""
    _lock = threading.Lock()
    risk_manager = types.SimpleNamespace(calc_risk=lambda *a, **k: 0.0)
    signal_params = types.SimpleNamespace(
        window=60,
        dynamic_quantile=0.8,
        base_th=0.0,
        low_base=0.0,
        quantile=0.8,
        rev_boost=0.0,
        rev_th_mult=1.0,
    )
    dynamic_th_params = DynamicThresholdParams()
    rsi_k = 1.0
    veto_conflict_count = 1
    all_scores_list: list[float] = []

    def __init__(self, cfg: RobustSignalGeneratorConfig | None = None):
        self.cfg = cfg

    def _make_cache_key(self, features: Mapping[str, Any], period: str):  # pragma: no cover
        return (period, tuple(sorted(features.items())))

    def get_feat_value(self, row: Mapping[str, Any], key: str, default: Any = 0):  # pragma: no cover
        return row.get(key, default)

    def generate_signal(self, features_1h, features_4h, features_d1, features_15m=None, **kwargs):
        return core.generate_signal(
            features_1h,
            features_4h,
            features_d1,
            features_15m,
            **kwargs,
        )

    def stop_weight_update_thread(self):  # pragma: no cover
        """Legacy no-op method kept for compatibility."""
        return None

    def detect_market_regime(self, *args, **kwargs):  # pragma: no cover
        return "range"

    # ------------------------------------------------------------------
    # Position finalization helpers
    # ------------------------------------------------------------------
    def _compute_vote(
        self,
        fused_score: float,
        ai_scores: Mapping[str, float],
        short_mom: float,
        vol_breakout: float | None,
        factor_scores: Mapping[str, float],
        score_details: Mapping[str, Any],
        confirm_15m: float,
        ob_imb: float,
        base_th: float,
    ) -> dict[str, Any]:
        """Compute vote using probabilistic model if available."""

        model_path: Path | None = None
        if self.cfg:
            mp = self.cfg.model_paths.get("voting_model")
            if isinstance(mp, Mapping):
                first = next(iter(mp.values()), None)
                if first is not None:
                    model_path = Path(first)
            elif isinstance(mp, str):
                model_path = Path(mp)

        model = safe_load(model_path or DEFAULT_MODEL_PATH)
        vote: float
        prob: float
        confidence: float

        if model is None:
            # fallback to legacy linear vote
            vote_weights = getattr(self, "vote_weights", {"ai": 1.0})
            raw_vote = vote_weights.get("ai", 1.0) * ai_scores.get("1h", 0.0)
            prob = 1.0 / (1.0 + math.exp(-raw_vote))
            confidence = abs(prob - 0.5) * 2.0
            vote = math.copysign(confidence, raw_vote)
        else:
            def sgn(x: float) -> int:
                return 1 if x > 0 else -1 if x < 0 else 0

            features = {
                "ai_dir": sgn(ai_scores.get("1h", 0.0)),
                "short_mom_dir": sgn(short_mom),
                "vol_breakout_dir": sgn(vol_breakout or 0.0),
                "trend_dir": sgn(factor_scores.get("trend", 0.0)),
                "confirm_dir": sgn(confirm_15m),
                "ob_dir": sgn(ob_imb),
                "abs_ai_score": abs(ai_scores.get("1h", 0.0)),
                "abs_momentum": abs(short_mom),
                "consensus_all": score_details.get("consensus_all", 0.0),
                "ic_weight": score_details.get("ic_weight", 0.0),
                "abs_score_minus_base_th": abs(fused_score) - base_th,
                "confirm_15m": confirm_15m,
                "short_mom": short_mom,
                "ob_imb": ob_imb,
            }
            X = [[features.get(col, 0.0) for col in model.feature_cols]]
            proba = model.predict_proba(X)[0]
            prob = float(proba[1])
            confidence = max(prob, 1 - prob)
            direction = 1 if prob >= 0.5 else -1
            vote = direction * confidence

        prob_margin = self.cfg.prob_margin if self.cfg else 0.0
        weak_vote = 0.5 - prob_margin <= prob <= 0.5 + prob_margin
        strong_prob_th = self.cfg.strong_prob_th if self.cfg else 1.0
        strong_confirm = confidence >= strong_prob_th

        return {
            "vote": float(vote),
            "prob": float(prob),
            "confidence": float(confidence),
            "weak_vote": weak_vote,
            "strong_confirm": strong_confirm,
        }

    def finalize_position(
        self,
        fused_score: float,
        risk_info: dict,
        logic_score: float,
        env_score: float,
        ai_scores: dict,
        fs: dict,
        scores: dict,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        std_15m: dict,
        raw_f1h: dict,
        raw_f4h: dict,
        raw_fd1: dict,
        raw_f15m: dict,
        vol_preds: dict,
        open_interest: dict | None,
        global_metrics: dict | None,
        *,
        short_mom: float = 0.0,
        ob_imb: float = 0.0,
        confirm_15m: float = 0.0,
        extreme_reversal: bool = False,
        cache: dict | None = None,
        symbol: str | None = None,
    ) -> dict | None:
        """Compute final trade parameters.

        该方法是旧版 :class:`RobustSignalGenerator` 的 ``finalize_position`` 的轻量化
        实现, 旨在满足单元测试对接口的依赖。逻辑上它会:

        1. 根据账户风险(当日亏损、CVaR)决定是否继续;
        2. 调用 :mod:`risk_filters` 获取分数/仓位乘数;
        3. 使用 ``calc_position_size`` 计算基础仓位并统一应用倍率;
        4. 根据风险预算与最小敞口约束得出最终仓位, 同时给出止盈止损价。
        """

        cache = cache or {}

        # ------------------------- risk pre-checks ----------------------
        account = getattr(self, "account", None)
        if account is not None:
            day_loss = getattr(account, "day_loss_pct", lambda: 0.0)()
            if day_loss > 0.03:
                return None
            alpha = getattr(self, "cvar_alpha", None)
            if alpha is not None and hasattr(account, "pnl_history"):
                if cvar_limit(getattr(account, "pnl_history"), alpha) < 0:
                    return None

        # -------------------------- risk filters -----------------------
        details = dict(risk_info.get("details", {}))
        score_mult = risk_info.get("score_mult")
        pos_mult = risk_info.get("pos_mult")
        if score_mult is None or pos_mult is None:
            sm, pm, reasons = self.risk_filters.apply_risk_filters(
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
                self.all_scores_list,
                risk_info.get("rev_dir", 0),
                cache,
                global_metrics,
                symbol,
                dyn_base=None,
            )
            score_mult = 1.0 if score_mult is None else score_mult
            pos_mult = 1.0 if pos_mult is None else pos_mult
            score_mult *= sm
            pos_mult *= pm
            if reasons:
                details.setdefault("penalties", []).extend(reasons)
            fused_for_pos = fused_score * sm
            final_score = fused_for_pos * score_mult / sm
        else:
            fused_for_pos = fused_score
            final_score = fused_score * score_mult

        # -------------------------- direction & vote --------------------
        vol_breakout = raw_f1h.get("vol_breakout_1h")
        if vol_breakout is None:
            vol_breakout = std_1h.get("vol_breakout_1h")
        vote_res = self._compute_vote(
            fused_for_pos,
            ai_scores,
            short_mom,
            vol_breakout,
            fs.get("1h", {}),
            scores,
            confirm_15m,
            ob_imb,
            risk_info.get("base_th", self.signal_params.base_th),
        )
        vote = vote_res["vote"]
        prob = vote_res["prob"]
        confidence = vote_res["confidence"]
        weak_vote = vote_res["weak_vote"]
        strong_confirm = vote_res["strong_confirm"]

        min_vote = getattr(self, "signal_filters", {}).get("min_vote", 0.0)
        direction = 1 if vote > 0 else -1 if vote < 0 else 0
        zero_reason = None
        if abs(vote) < min_vote or (vol_breakout is not None and vol_breakout <= 0):
            direction = 0
            zero_reason = RiskReason.VOTE_FILTER.value

        # record some vote related details
        details.setdefault("vote", {}).update(
            {
                "value": vote,
                "prob": prob,
                "confidence": confidence,
                "weak_vote": weak_vote,
                "ob_th": getattr(self, "ob_th_params", {"min_ob_th": 0}).get(
                    "min_ob_th", 0
                ),
            }
        )
        details["strong_confirm_vote"] = strong_confirm

        # ----------------------- position sizing -----------------------
        base_th = risk_info.get("base_th", self.signal_params.base_th)
        gamma = self.signal_threshold_cfg.get("gamma", getattr(self.signal_params, "gamma", 0.05))
        min_pos = self.signal_threshold_cfg.get(
            "min_pos", getattr(self.signal_params, "min_pos", 0.0)
        )

        pos_size = calc_position_size(
            fused_for_pos,
            base_th,
            max_position=self.max_position,
            gamma=gamma,
            cvar_target=risk_info.get("cvar_target"),
            vol_target=risk_info.get("vol_target"),
            min_exposure=min_pos,
        )

        pos_size *= pos_mult

        vol_ratio = raw_f1h.get("vol_ma_ratio_1h")
        if vol_ratio is None:
            vol_ratio = std_1h.get("vol_ma_ratio_1h")

        factors = {
            "low_volume": {
                "rsg": self,
                "regime": risk_info.get("regime", "range"),
                "vol_ratio": vol_ratio,
                "fused_score": fused_for_pos,
                "base_th": base_th,
                "consensus_all": False,
            },
            "vol_prediction": vol_preds.get("1h"),
        }
        pos_size, flags = apply_normalized_multipliers(pos_size, factors)
        if flags.get("low_volume"):
            zero_reason = RiskReason.VOL_RATIO.value
            details.setdefault("penalties", []).append(RiskReason.VOL_RATIO.value)

        if direction == 0:
            pos_size = 0.0
        else:
            pos_size = max(min_pos, min(pos_size, self.max_position))

        price = raw_f1h.get("close")
        atr = raw_f1h.get("atr_pct_1h")
        if hasattr(self, "position_sizer") and hasattr(
            self.position_sizer, "compute_tp_sl"
        ):
            tp, sl = self.position_sizer.compute_tp_sl(
                price, atr, direction, regime=risk_info.get("regime")
            )
        else:  # pragma: no cover - fallback
            from .signal.position_sizing import compute_tp_sl as _compute_tp_sl

            tp, sl = _compute_tp_sl(
                self, price, atr, direction, regime=risk_info.get("regime")
            )

        # 风险预算限制
        rb = self.cfg.get("risk_budget_per_trade") if self.cfg else None
        if (
            rb
            and account is not None
            and price is not None
            and sl is not None
            and abs(price - sl) > 0
        ):
            cap = rb * getattr(account, "equity", 1.0) / abs(price - sl)
            pos_size = min(pos_size, cap)

        max_pct = self.cfg.get("max_pos_pct") if self.cfg else None
        if max_pct is not None:
            pos_size = min(pos_size, max_pct)

        if pos_size == 0 and zero_reason is None:
            penalties = details.get("penalties", [])
            zero_reason = ",".join(penalties) if penalties else RiskReason.NO_DIRECTION.value

        result = {
            "position_size": float(pos_size),
            "signal": int(direction),
            "score": float(final_score),
            "take_profit": tp,
            "stop_loss": sl,
            "zero_reason": zero_reason,
            "details": details,
            "vote": float(vote),
            "prob": float(prob),
            "weak_vote": weak_vote,
            "strong_confirm": strong_confirm,
        }

        if getattr(self, "enable_factor_breakdown", False):
            fb = {k: fs.get("1h", {}).get(k, 0.0) for k in fs.get("1h", {})}
            fb["ai"] = ai_scores.get("1h", 0.0)
            result["factor_breakdown"] = fb

        return result


__all__ = ["RobustSignalGenerator", "RobustSignalGeneratorConfig"]

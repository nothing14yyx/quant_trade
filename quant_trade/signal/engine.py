"""High level orchestration of signal generation pipeline.

该模块提供 :class:`SignalEngine`，用于将若干组件协同在一起运行。
"""

from __future__ import annotations

from typing import Any, Mapping

from .core import RobustSignalGenerator
from .factor_scorer import FactorScorerImpl
from .fusion_rule import FusionRuleBased
from .position_sizer import PositionSizerImpl
from .predictor_adapter import PredictorAdapter
from .risk_filters import RiskFiltersImpl


def _to_float_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """将结果字典转换为包含 ``float`` 等类型的字典。

    主要数值字段会被转换为 ``float``，其余字段保持原样，便于
    测试读取 ``details`` 等诊断信息。
    """

    tp = data.get("take_profit")
    sl = data.get("stop_loss")
    res: dict[str, Any] = {
        "signal": float(data.get("signal", 0.0)),
        "score": float(data.get("score", 0.0)),
        "position_size": float(data.get("position_size", 0.0)),
        "take_profit": float(tp) if tp is not None else None,
        "stop_loss": float(sl) if sl is not None else None,
    }
    for key, value in data.items():
        if key not in res:
            res[key] = value
    return res


class SignalEngine:
    """封装信号生成流程的引擎类。

    Parameters
    ----------
    rsg:
        :class:`RobustSignalGenerator` 实例，提供底层辅助方法。
    predictor:
        :class:`PredictorAdapter`，用于计算 AI 分数。
    factor_scorer:
        :class:`FactorScorerImpl`，因子得分计算器。
    fusion_rule:
        :class:`FusionRuleBased`，多周期融合规则实现。
    risk_filters:
        :class:`RiskFiltersImpl`，风险与阈值过滤器。
    position_sizer:
        :class:`PositionSizerImpl`，仓位及 TP/SL 计算器。
    """

    def __init__(
        self,
        rsg: RobustSignalGenerator,
        predictor: PredictorAdapter,
        factor_scorer: FactorScorerImpl,
        fusion_rule: FusionRuleBased,
        risk_filters: RiskFiltersImpl,
        position_sizer: PositionSizerImpl,
    ) -> None:
        self.rsg = rsg
        self.predictor = predictor
        self.factor_scorer = factor_scorer
        self.fusion_rule = fusion_rule
        self.risk_filters = risk_filters
        self.position_sizer = position_sizer

        # 确保 RobustSignalGenerator 使用外部传入的组件
        self.rsg.predictor = predictor
        self.rsg.factor_scorer = factor_scorer
        self.rsg.fusion_rule = fusion_rule
        self.rsg.risk_filters = risk_filters
        self.rsg.position_sizer = position_sizer

    # ------------------------------------------------------------------
    def run(self, ctx: Mapping[str, Any]) -> dict[str, float] | None:
        """执行一次信号计算并返回结果字典。

        ``ctx`` 需要提供特征、原始特征、盘口不平衡、全局指标等键值，
        具体字段与 :meth:`RobustSignalGenerator.generate_signal` 相同。
        该方法内部依次调用 ``_prepare_inputs``、``_compute_scores``、
        ``_risk_checks`` 与 ``_calc_position_and_sl_tp`` 等步骤，最终返回
        包含 ``signal``、``score``、``position_size``、``take_profit`` 和
        ``stop_loss`` 的简化结果字典。
        """

        prepared = self.rsg._prepare_inputs(
            ctx.get("features_1h"),
            ctx.get("features_4h"),
            ctx.get("features_d1"),
            ctx.get("features_15m"),
            ctx.get("raw_features_1h"),
            ctx.get("raw_features_4h"),
            ctx.get("raw_features_d1"),
            ctx.get("raw_features_15m"),
            ctx.get("order_book_imbalance"),
            ctx.get("symbol"),
        )

        scores = self.rsg._compute_scores(
            prepared["pf_1h"],
            prepared["pf_4h"],
            prepared["pf_d1"],
            prepared["pf_15m"],
            prepared["deltas"],
            ctx.get("global_metrics"),
            ctx.get("open_interest"),
            prepared["ob_imb"],
            ctx.get("symbol"),
        )
        if scores is None:
            return None

        fused_score = scores["fused_score"]
        logic_score = scores["logic_score"]
        env_score = scores["env_score"]
        fs = scores["fs"]
        score_details = scores["scores"]
        score_details["conflict"] = scores.get("conflict")
        ai_scores = score_details["ai_scores"]
        vol_preds = score_details["vol_preds"]
        rise_preds = score_details["rise_preds"]
        drawdown_preds = score_details["drawdown_preds"]
        short_mom = score_details["short_mom"]
        confirm_15m = score_details["confirm_15m"]
        extreme_reversal = score_details["extreme_reversal"]

        std_1h = prepared["std_1h"]
        std_4h = prepared["std_4h"]
        std_d1 = prepared["std_d1"]
        std_15m = prepared["std_15m"]
        raw_f1h = prepared["raw_f1h"]
        raw_f4h = prepared["raw_f4h"]
        raw_fd1 = prepared["raw_fd1"]
        raw_f15m = prepared["raw_f15m"]
        cache = prepared["cache"]
        rev_dir = prepared["rev_dir"]
        ob_imb = score_details["ob_imb"]
        ts = prepared["ts"]

        phase = getattr(self.rsg, "market_phase", "range")
        if isinstance(phase, dict):
            phase = phase.get("phase", "range")
        mults = getattr(self.rsg, "phase_dir_mult", {})
        if fused_score > 0:
            fused_score *= mults.get("long", 1.0)
        elif fused_score < 0:
            fused_score *= mults.get("short", 1.0)

        pre_res, direction, _ = self.rsg._precheck_and_direction(
            fused_score,
            std_1h,
            std_4h,
            std_d1,
            std_15m,
            raw_f1h,
            raw_f4h,
            raw_fd1,
            raw_f15m,
            ai_scores,
            fs,
            score_details,
            score_details["local_details"],
            score_details["consensus_all"],
            score_details["consensus_14"],
            vol_preds,
            rise_preds,
            drawdown_preds,
            confirm_15m,
            cache,
        )
        if pre_res is not None:
            return _to_float_dict(pre_res)

        risk_info = self.rsg._risk_checks(
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
            ctx.get("open_interest"),
            ctx.get("all_scores_list"),
            rev_dir,
            cache,
            ctx.get("global_metrics"),
            ctx.get("symbol"),
        )
        if risk_info is None:
            return None

        result = self.rsg._calc_position_and_sl_tp(
            risk_info["fused_score"],
            risk_info.get("pos_mult", 1.0),
            risk_info.get("details", {}).get("penalties", []),
            risk_info,
            logic_score,
            env_score,
            ai_scores,
            fs,
            score_details,
            std_1h,
            std_4h,
            std_d1,
            std_15m,
            raw_f1h,
            raw_f4h,
            raw_fd1,
            raw_f15m,
            vol_preds,
            rise_preds,
            drawdown_preds,
            short_mom,
            ob_imb,
            confirm_15m,
            extreme_reversal,
            cache,
            ctx.get("symbol"),
            ts,
        )
        return _to_float_dict(result)

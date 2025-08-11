from __future__ import annotations

"""High level orchestration of signal generation pipeline.

该模块提供 :class:`SignalEngine`，用于将若干组件协同在一起运行。
"""

from typing import Any, Mapping

from .core import RobustSignalGenerator
from .predictor_adapter import PredictorAdapter
from .factor_scorer import FactorScorerImpl
from .fusion_rule import FusionRuleBased
from .risk_filters import RiskFiltersImpl
from .position_sizer import PositionSizerImpl


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
    def run(self, ctx: Mapping[str, Any]):
        """执行一次信号计算并返回结果字典。

        ``ctx`` 参数与 :meth:`RobustSignalGenerator.generate_signal` 一致，
        以字典形式提供所需输入数据。该方法内部依次调用
        ``_prepare_inputs``、``_compute_scores``、``_risk_checks`` 与
        ``_calc_position_and_sl_tp`` 等步骤，因而最终的返回结果和日志
        与旧实现保持一致。
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
            return pre_res

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
        return result

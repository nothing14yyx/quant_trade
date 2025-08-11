"""Signal generation subpackage exports."""

from .core import (
    SignalThresholdParams,
    DynamicThresholdParams,
    RobustSignalGeneratorConfig,
    PeriodFeatures,
    RobustSignalGenerator,
    DEFAULT_AI_DIR_EPS,
    DEFAULT_POS_K_RANGE,
    DEFAULT_POS_K_TREND,
    DEFAULT_LOW_BASE,
    DEFAULT_LOW_VOL_RATIO,
    DEFAULT_CACHE_MAXSIZE,
    DEFAULTS,
    SAFE_FALLBACKS,
)
from .thresholding_dynamic import (
    ThresholdingDynamic,
    DynamicThresholdInput,
    compute_dynamic_threshold,
)
from .utils import (
    softmax,
    sigmoid,
    smooth_score,
    smooth_series,
    weighted_quantile,
    _calc_history_base,
    risk_budget_threshold,
    adjust_score,
    volume_guard,
    cap_positive,
    fused_to_risk,
    sigmoid_dir,
    sigmoid_confidence,
)
from .voting_model import VotingModel, load_cached_model
from .factor_scorer import FactorScorerImpl
from .fusion_rule import FusionRuleBased
from .risk_filters import RiskFiltersImpl

__all__ = [
    "SignalThresholdParams",
    "DynamicThresholdParams",
    "RobustSignalGeneratorConfig",
    "DynamicThresholdInput",
    "PeriodFeatures",
    "RobustSignalGenerator",
    "ThresholdingDynamic",
    "compute_dynamic_threshold",
    "DEFAULT_AI_DIR_EPS",
    "DEFAULT_POS_K_RANGE",
    "DEFAULT_POS_K_TREND",
    "DEFAULT_LOW_BASE",
    "DEFAULT_LOW_VOL_RATIO",
    "DEFAULT_CACHE_MAXSIZE",
    "DEFAULTS",
    "SAFE_FALLBACKS",
    "softmax",
    "sigmoid",
    "smooth_score",
    "smooth_series",
    "weighted_quantile",
    "_calc_history_base",
    "risk_budget_threshold",
    "adjust_score",
    "volume_guard",
    "cap_positive",
    "fused_to_risk",
    "sigmoid_dir",
    "sigmoid_confidence",
    "VotingModel",
    "load_cached_model",
    "FactorScorerImpl",
    "FusionRuleBased",
    "RiskFiltersImpl",
]

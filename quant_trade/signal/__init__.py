"""Signal generation subpackage exports."""

from .core import generate_signal
from .features_to_scores import get_factor_scores, get_factor_scores_batch
from .ai_inference import get_period_ai_scores, get_reg_predictions
from .multi_period_fusion import fuse_scores
from .dynamic_thresholds import (
    DynamicThresholdInput,
    SignalThresholdParams,
    DynamicThresholdParams,
    ThresholdingDynamic,
    calc_dynamic_threshold,
)
from .robust_signal_generator import compute_dynamic_threshold
from .thresholding_dynamic import ThresholdParams
from .predictor_adapter import PredictorAdapter
from .factor_scorer import FactorScorerImpl
from .fusion_rule import FusionRuleBased
from .risk_filters import RiskFiltersImpl
from .position_sizer import PositionSizerImpl
from .position_sizing import calc_position_size
from .voting_model import VotingModel
from .vote_fusion import Vote, fuse_votes

try:  # optional
    from .risk_filters import compute_risk_multipliers
except Exception:  # pragma: no cover
    compute_risk_multipliers = None

__all__ = [
    "generate_signal",
    "get_factor_scores",
    "get_factor_scores_batch",
    "get_period_ai_scores",
    "get_reg_predictions",
    "fuse_scores",
    "DynamicThresholdInput",
    "SignalThresholdParams",
    "DynamicThresholdParams",
    "ThresholdingDynamic",
    "PredictorAdapter",
    "FactorScorerImpl",
    "FusionRuleBased",
    "RiskFiltersImpl",
    "PositionSizerImpl",
    "compute_dynamic_threshold",
    "calc_dynamic_threshold",
    "ThresholdParams",
    "calc_position_size",
    "VotingModel",
    "Vote",
    "fuse_votes",
]
if compute_risk_multipliers is not None:
    __all__.append("compute_risk_multipliers")


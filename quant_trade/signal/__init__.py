"""Signal generation subpackage exports."""

from .core import generate_signal
from .features_to_scores import get_factor_scores
from .ai_inference import get_period_ai_scores, get_reg_predictions
from .multi_period_fusion import fuse_scores
from .dynamic_thresholds import DynamicThresholdInput, calc_dynamic_threshold
from .position_sizing import calc_position_size

try:  # optional
    from .risk_filters import compute_risk_multipliers
except Exception:  # pragma: no cover
    compute_risk_multipliers = None

__all__ = [
    "generate_signal",
    "get_factor_scores",
    "get_period_ai_scores",
    "get_reg_predictions",
    "fuse_scores",
    "DynamicThresholdInput",
    "calc_dynamic_threshold",
    "calc_position_size",
]
if compute_risk_multipliers is not None:
    __all__.append("compute_risk_multipliers")


# Import key modules for convenient access
from . import robust_signal_generator

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from . import data_loader, feature_engineering

# Re-export commonly used classes
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .robust_signal_generator import RobustSignalGenerator
from .coinmetrics_loader import CoinMetricsLoader
from .offline_price_table import generate_offline_price_table
from .constants import ZeroReason


def __getattr__(name: str):
    if name in {"data_loader", "feature_engineering"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "DataLoader",
    "FeatureEngineer",
    "RobustSignalGenerator",
    "CoinMetricsLoader",
    "ZeroReason",
    "generate_offline_price_table",
]

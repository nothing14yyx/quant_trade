"""Package convenience exports with lazy loading to avoid heavy dependencies."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from . import data_loader, feature_engineering
    from .data_loader import DataLoader
    from .feature_engineering import FeatureEngineer
    from .robust_signal_generator import RobustSignalGenerator
    from .coinmetrics_loader import CoinMetricsLoader
    from .constants import RiskReason, ZeroReason
    from .offline_price_table import generate_offline_price_table

__all__ = [
    "DataLoader",
    "FeatureEngineer",
    "RobustSignalGenerator",
    "CoinMetricsLoader",
    "RiskReason",
    "ZeroReason",
    "generate_offline_price_table",
]


def __getattr__(name: str):
    if name in {"data_loader", "feature_engineering"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in __all__:
        module_map = {
            "DataLoader": "data_loader",
            "FeatureEngineer": "feature_engineering",
            "RobustSignalGenerator": "robust_signal_generator",
            "CoinMetricsLoader": "coinmetrics_loader",
            "RiskReason": "constants",
            "ZeroReason": "constants",
            "generate_offline_price_table": "offline_price_table",
        }
        module = importlib.import_module(f".{module_map[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__} has no attribute {name}")

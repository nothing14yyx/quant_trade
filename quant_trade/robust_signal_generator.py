"""Backward-compatible wrappers for the new signal pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import threading
import types

from .signal import core
from .signal.dynamic_thresholds import DynamicThresholdParams, SignalThresholdParams


@dataclass
class RobustSignalGeneratorConfig:
    """Minimal configuration placeholder for tests."""

    model_paths: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    feature_cols_1h: list[str] = field(default_factory=list)
    feature_cols_4h: list[str] = field(default_factory=list)
    feature_cols_d1: list[str] = field(default_factory=list)

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


__all__ = ["RobustSignalGenerator", "RobustSignalGeneratorConfig"]

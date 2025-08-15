
import yaml
import logging
from pathlib import Path

import pytest

from quant_trade.config_schema import SignalConfig
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.risk_manager import RiskManager


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def _make_rsg(tmp_path: Path, cfg: dict):
    cfg_path = _write_cfg(tmp_path, cfg)
    init_cfg = {
        "models": {},
        "feature_cols": {
            "1h": ["rsi_1h"],
            "4h": ["rsi_4h"],
            "d1": ["rsi_d1"],
        },
        "enable_ai": False,
    }
    config_obj = RobustSignalGeneratorConfig.from_cfg(init_cfg, cfg_path)
    rsg = RobustSignalGenerator(config_obj)
    rsg.risk_manager = RiskManager()
    return rsg


def test_cfg_validation_success(tmp_path):
    cfg = SignalConfig().model_dump()
    rsg = _make_rsg(tmp_path, cfg)
    assert rsg.cfg == cfg


def test_cfg_validation_warning(tmp_path, caplog):
    cfg = SignalConfig().model_dump()
    cfg["penalty_factor"] = "bad"
    with caplog.at_level(logging.WARNING):
        rsg = _make_rsg(tmp_path, cfg)
    assert "Config validation failed" in caplog.text
    assert rsg.cfg["penalty_factor"] == "bad"


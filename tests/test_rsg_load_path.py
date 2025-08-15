import pytest
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
from quant_trade.risk_manager import RiskManager


def test_model_path_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_paths = {"1h": {"cls": "models/model_1h_cls.pkl"}}
    cfg = RobustSignalGeneratorConfig(
        model_paths=model_paths,
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
    )
    rsg = RobustSignalGenerator(cfg)
    rsg.risk_manager = RiskManager()
    assert "cls" in rsg.models.get("1h", {})
    assert hasattr(rsg.models["1h"]["cls"]["pipeline"], "predict")
    rsg.update_weights()


def test_base_weights_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
ic_scores:
  base_weights:
    ai: 1
    trend: 1
    momentum: 2
    volatility: 2
    volume: 1
    sentiment: 1
    funding: 3
""",
        encoding="utf-8",
    )
    cfg = RobustSignalGeneratorConfig(
        model_paths={},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
        config_path=cfg_path,
    )
    rsg = RobustSignalGenerator(cfg)
    rsg.risk_manager = RiskManager()
    total = 1 + 1 + 2 + 2 + 1 + 1 + 3
    expected = {
        "ai": 1 / total,
        "trend": 1 / total,
        "momentum": 2 / total,
        "volatility": 2 / total,
        "volume": 1 / total,
        "sentiment": 1 / total,
        "funding": 3 / total,
    }
    assert rsg.base_weights == pytest.approx(expected)
    rsg.update_weights()


def test_signal_threshold_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
signal_threshold:
  base_th: 0.2
""",
        encoding="utf-8",
    )
    cfg = RobustSignalGeneratorConfig(
        model_paths={},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
        config_path=cfg_path,
    )
    rsg = RobustSignalGenerator(cfg)
    rsg.risk_manager = RiskManager()
    assert rsg.signal_threshold_cfg["base_th"] == pytest.approx(0.2)
    rsg.update_weights()


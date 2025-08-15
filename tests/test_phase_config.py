import yaml
import pytest
from quant_trade.robust_signal_generator import RobustSignalGenerator, RobustSignalGeneratorConfig
from quant_trade.risk_manager import RiskManager


def test_update_market_phase_uses_config(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yml"
    cfg = {
        "market_phase": {
            "phase_th_mult": {"bull": 0.8, "bear": 1.2, "range": 1.0}
        }
    }
    yaml.safe_dump(cfg, cfg_path.open("w"))
    rsg_cfg = RobustSignalGeneratorConfig(
        model_paths={},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
        config_path=cfg_path,
    )
    rsg = RobustSignalGenerator(rsg_cfg)
    rsg.risk_manager = RiskManager()
    monkeypatch.setattr(
        "quant_trade.market_phase.get_market_phase", lambda engine: {"phase": "bull"}
    )
    rsg.update_market_phase(None)
    assert rsg.phase_th_mult == pytest.approx(0.8)
    rsg.update_weights()


def test_update_phase_dir_mult(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yml"
    cfg = {
        "market_phase": {
            "phase_dir_mult": {
                "bull": {"long": 2, "short": 1},
                "bear": {"long": 1, "short": 2},
                "range": {"long": 1, "short": 1},
            }
        }
    }
    yaml.safe_dump(cfg, cfg_path.open("w"))
    rsg_cfg = RobustSignalGeneratorConfig(
        model_paths={},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
        config_path=cfg_path,
    )
    rsg = RobustSignalGenerator(rsg_cfg)
    rsg.risk_manager = RiskManager()
    monkeypatch.setattr(
        "quant_trade.market_phase.get_market_phase", lambda engine: {"phase": "bear"}
    )
    rsg.update_market_phase(None)
    assert rsg.phase_dir_mult == {"long": 1, "short": 2}
    rsg.update_weights()

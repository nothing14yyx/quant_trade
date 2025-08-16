import numpy as np
import pytest
import yaml
from pathlib import Path

from quant_trade.signal.decision import DecisionConfig, decide_signal


def test_decide_signal_basic():
    cfg = DecisionConfig(p_up_min=0.6, p_down_min=0.6, margin_min=0.1, kelly_gamma=0.5, w_max=1.0)
    res_buy = decide_signal(np.array([0.1, 0.2, 0.7]), None, None, None, False, cfg)
    assert res_buy["action"] == "BUY"
    assert res_buy["size"] > 0

    res_sell = decide_signal(np.array([0.7, 0.2, 0.1]), None, None, None, False, cfg)
    assert res_sell["action"] == "SELL"

    res_hold = decide_signal(np.array([0.4, 0.2, 0.4]), None, None, None, False, cfg)
    assert res_hold["action"] == "HOLD"


def test_vol_pred_adjusts_threshold():
    cfg = DecisionConfig(p_up_min=0.6, p_down_min=0.6, margin_min=0.05, kelly_gamma=0.5, w_max=1.0)
    # Without vol adjustment this would trigger BUY, but high vol raises threshold
    res = decide_signal(np.array([0.35, 0.65]), None, None, 0.1, False, cfg)
    assert res["action"] == "HOLD"


def test_size_weight_with_predictions():
    cfg = DecisionConfig(p_up_min=0.6, p_down_min=0.6, margin_min=0.1, kelly_gamma=1.0, w_max=1.0)
    res = decide_signal(
        np.array([0.1, 0.1, 0.8]), 0.2, 0.05, None, False, cfg
    )
    assert res["action"] == "BUY"
    assert res["size"] == pytest.approx(0.03, rel=1e-3)
    assert "rise=0.20" in res["note"] and "dd=0.05" in res["note"]


def test_decision_config_from_yaml():
    path = Path("quant_trade/utils/config.yaml")
    cfg = yaml.safe_load(path.read_text())
    dcfg = DecisionConfig.from_dict(cfg.get("signal", {}))
    assert isinstance(dcfg, DecisionConfig)

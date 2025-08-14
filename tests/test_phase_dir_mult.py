import yaml
import pytest
from collections import deque
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)


def setup_simple_rsg(tmp_path, monkeypatch, phase, mult_cfg, score):
    cfg_path = tmp_path / "cfg.yml"
    yaml.safe_dump({"market_phase": mult_cfg}, cfg_path.open("w"))
    rsg_cfg = RobustSignalGeneratorConfig(
        model_paths={},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
        config_path=cfg_path,
    )
    rsg = RobustSignalGenerator(rsg_cfg)
    monkeypatch.setattr(
        "quant_trade.market_phase.get_market_phase",
        lambda engine: {"phase": phase},
    )
    rsg.update_market_phase(None)

    rsg._prepare_inputs = lambda *a, **k: {
        "pf_1h": {},
        "pf_4h": {},
        "pf_d1": {},
        "pf_15m": {},
        "deltas": {},
        "ob_imb": 0,
        "std_1h": {},
        "std_4h": {},
        "std_d1": {},
        "std_15m": {},
        "raw_f1h": {},
        "raw_f4h": {},
        "raw_fd1": {},
        "raw_f15m": {},
        "ts": 0,
        "cache": {
            "history_scores": deque(),
            "oi_change_history": deque(),
            "_raw_history": {"1h": deque()},
        },
        "rev_dir": 0,
    }

    rsg._compute_scores = lambda *a, **k: {
        "fused_score": score,
        "logic_score": 0.0,
        "env_score": 0.0,
        "risk_score": 0.0,
        "fs": {"1h": {}, "4h": {}, "d1": {}},
        "scores": {
            "local_details": {},
            "consensus_all": False,
            "consensus_14": False,
            "consensus_4d1": False,
            "short_mom": 0,
            "confirm_15m": 0,
            "oi_overheat": False,
            "th_oi": 0,
            "oi_chg": 0,
            "ob_imb": 0,
            "ai_scores": {"1h": 0, "4h": 0, "d1": 0},
            "vol_preds": {},
            "rise_preds": {},
            "drawdown_preds": {},
            "extreme_reversal": False,
        },
    }

    rsg._precheck_and_direction = lambda *a, **k: (None, int(score > 0) - int(score < 0), False)
    rsg._risk_checks = lambda *a, **k: {"fused_score": a[0], "base_th": 0.1}
    captured = {}

    def fake_calc(fs, *args, **kwargs):
        captured["score"] = fs
        return {
            "signal": int(fs > 0) - int(fs < 0),
            "score": fs,
            "position_size": 1.0,
        }

    rsg._calc_position_and_sl_tp = fake_calc
    return rsg, captured


def test_bull_phase_dir_multiplier(tmp_path, monkeypatch):
    cfg = {
        "phase_dir_mult": {
            "bull": {"long": 2.0, "short": 0.5},
            "bear": {"long": 0.5, "short": 2.0},
            "range": {"long": 1.0, "short": 1.0},
        }
    }
    rsg, cap = setup_simple_rsg(tmp_path, monkeypatch, "bull", cfg, 0.5)
    res = rsg.generate_signal({}, {}, {})
    assert cap["score"] == pytest.approx(1.0)
    assert res["signal"] == 1
    rsg.update_weights()


def test_bear_phase_dir_multiplier(tmp_path, monkeypatch):
    cfg = {
        "phase_dir_mult": {
            "bull": {"long": 2.0, "short": 0.5},
            "bear": {"long": 0.5, "short": 2.0},
            "range": {"long": 1.0, "short": 1.0},
        }
    }
    rsg, cap = setup_simple_rsg(tmp_path, monkeypatch, "bear", cfg, -0.5)
    res = rsg.generate_signal({}, {}, {})
    assert cap["score"] == pytest.approx(-1.0)
    assert res["signal"] == -1
    rsg.update_weights()


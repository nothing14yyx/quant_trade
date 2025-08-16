import yaml
import pytest

from quant_trade.signal import core


def _setup_stubs(monkeypatch):
    factor_scores = {"1h": {"trend": 1.0}, "4h": {}, "d1": {}}
    ai_scores = {"1h": 1.0, "4h": 0.0, "d1": 0.0}
    monkeypatch.setattr(core.features_to_scores, "get_factor_scores", lambda feats, period: factor_scores[period])
    monkeypatch.setattr(core.ai_inference, "get_period_ai_scores", lambda *a, **k: ai_scores)
    monkeypatch.setattr(core.ai_inference, "get_reg_predictions", lambda *a, **k: (0.0, 0.0, 0.0))
    monkeypatch.setattr(core.multi_period_fusion, "fuse_scores", lambda combined, *a, **k: (combined["1h"], 0, 0, 0))
    monkeypatch.setattr(core.dynamic_thresholds, "calc_dynamic_threshold", lambda inp: (0.0, 0.0))
    monkeypatch.setattr(core.risk_filters, "compute_risk_multipliers", lambda *a, **k: (1.0, 1.0, [], {}))
    monkeypatch.setattr(core.position_sizing, "calc_position_size", lambda *a, **k: 0.0)


def test_refresh_weights_interface(tmp_path, monkeypatch):
    _setup_stubs(monkeypatch)
    cache_file = tmp_path / "w_cache.yaml"
    monkeypatch.setattr(core, "_WEIGHT_CACHE_PATH", cache_file)
    monkeypatch.setattr(core, "_FLUSH_INTERVAL", 0)
    monkeypatch.setattr(core, "_last_weight_flush", 0)

    core.generate_signal({}, {}, {}, factor_weights={"trend": 1.0}, ic_stats={"ai_1h": 0.1, "factor_1h": 0.1, "trend": 0.2}, ic_threshold=0.2)

    assert cache_file.exists()
    data = yaml.safe_load(cache_file.read_text())
    assert data["w_ai"]["1h"] != 1.0

    cache_file.write_text(yaml.safe_dump({"w_ai": {"1h": 0.2}, "w_factor": {"1h": 0.3}, "category_ic": {"trend": 0.4}}))
    core.refresh_weights(cache_file)
    assert core._cached_w_ai["1h"] == pytest.approx(0.2)
    assert core._cached_w_factor["1h"] == pytest.approx(0.3)
    assert core.features_to_scores.category_ic["trend"] == pytest.approx(0.4)


def test_generate_signal_backtest_returns(monkeypatch):
    _setup_stubs(monkeypatch)
    monkeypatch.setattr(core, "_cached_w_ai", {"1h": 1.0, "4h": 1.0, "d1": 1.0})
    monkeypatch.setattr(core, "_cached_w_factor", {"1h": 1.0, "4h": 1.0, "d1": 1.0})
    monkeypatch.setattr(core, "_maybe_flush_weights", lambda *a, **k: None)

    res = core.generate_signal({}, {}, {}, factor_weights={"trend": 1.0}, backtest_returns={"ai_1h": 0.1, "factor_1h": 0.2, "trend": 0.1})
    assert res["score"] == pytest.approx(2.3)
    assert core._cached_w_ai["1h"] == pytest.approx(1.1)
    assert core._cached_w_factor["1h"] == pytest.approx(1.2)

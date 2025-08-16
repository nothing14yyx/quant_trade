import pytest

from quant_trade.signal import core, features_to_scores, ai_inference, multi_period_fusion, dynamic_thresholds, risk_filters, position_sizing


def test_generate_signal_uses_dynamic_min_exposure(monkeypatch):
    """generate_signal 应根据波动率调整最小敞口"""

    monkeypatch.setattr(features_to_scores, "get_factor_scores", lambda feats, period: {"x": 0.0})
    monkeypatch.setattr(
        ai_inference,
        "get_period_ai_scores",
        lambda predictor, pf, models, calibrators, cache=None: {"1h": 0.0, "4h": 0.0, "d1": 0.0},
    )
    monkeypatch.setattr(
        ai_inference,
        "get_reg_predictions",
        lambda predictor, pf, models: ({"1h": 0.8, "4h": 0.8, "d1": 0.8}, {}, {}),
    )
    monkeypatch.setattr(
        multi_period_fusion,
        "fuse_scores",
        lambda *a, **k: (0.05, False, False, False),
    )

    class DummyInput:
        pass

    monkeypatch.setattr(dynamic_thresholds, "DynamicThresholdInput", lambda *a, **k: DummyInput())
    monkeypatch.setattr(dynamic_thresholds, "calc_dynamic_threshold", lambda input: (0.1, 0.0))
    monkeypatch.setattr(risk_filters, "compute_risk_multipliers", lambda *a, **k: (1.0, 1.0, [], {}))

    captured = {}

    def fake_calc(fused_score, base_th, *, max_position, min_exposure, **kwargs):
        captured["min_exposure"] = min_exposure
        return 0.0

    monkeypatch.setattr(position_sizing, "calc_position_size", fake_calc)

    core.generate_signal({}, {}, {})

    assert captured["min_exposure"] == pytest.approx(0.2 * (1 - 0.8))

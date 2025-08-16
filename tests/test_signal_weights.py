import pytest

from quant_trade.signal import core


def test_generate_signal_weighting(monkeypatch):
    factor_scores = {"1h": {"trend": 1.0, "momentum": -1.0}, "4h": {}, "d1": {}}
    ai_scores = {"1h": 2.0, "4h": 0.0, "d1": 0.0}

    monkeypatch.setattr(core.features_to_scores, "get_factor_scores", lambda feats, period: factor_scores[period])
    monkeypatch.setattr(core.ai_inference, "get_period_ai_scores", lambda predictor, period_features, models, calibrators, cache=None: ai_scores)
    monkeypatch.setattr(core.ai_inference, "get_reg_predictions", lambda predictor, period_features, models: (0.0, 0.0, 0.0))
    monkeypatch.setattr(core.multi_period_fusion, "fuse_scores", lambda combined, *a, **k: (combined["1h"], 0, 0, 0))
    monkeypatch.setattr(core.dynamic_thresholds, "calc_dynamic_threshold", lambda inp: (0.0, 0.0))
    monkeypatch.setattr(core.risk_filters, "compute_risk_multipliers", lambda *a, **k: (1.0, 1.0, [], {}))
    monkeypatch.setattr(core.position_sizing, "calc_position_size", lambda *a, **k: 0.0)

    recorded = {}
    monkeypatch.setattr(core.features_to_scores, "record_ic", lambda stats: recorded.update({k: v for k, v in stats.items() if v is not None}) if stats else None)

    res = core.generate_signal(
        {}, {}, {},
        predictor=None,
        models=None,
        calibrators=None,
        w_ai={"1h": 1.0, "4h": 1.0, "d1": 1.0},
        w_factor=1.0,
        factor_weights={"trend": 1.0, "momentum": 1.0},
        ic_stats={"ai_1h": 0.1, "factor_1h": 0.15, "trend": 0.1, "momentum": 0.2},
        ic_threshold=0.2,
    )

    assert res["score"] == pytest.approx(0.75)
    assert recorded == {"trend": 0.1, "momentum": 0.2}

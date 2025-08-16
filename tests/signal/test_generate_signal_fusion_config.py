from quant_trade.signal import core


def test_generate_signal_reads_fusion_config(monkeypatch):
    cfg = {
        "fusion": {
            "min_agree": 3,
            "conflict_mult": 0.5,
            "cycle_weight": {
                "strong": 1.2,
                "weak": 0.8,
                "opposite": 0.6,
                "conflict": 0.4,
            },
        }
    }

    class DummyCM:
        def __init__(self, path):
            pass

        def get(self, key, default=None):
            return cfg.get(key, default)

    monkeypatch.setattr(core, "ConfigManager", DummyCM)
    monkeypatch.setattr(core.features_to_scores, "get_factor_scores", lambda feats, period: {})
    monkeypatch.setattr(
        core.ai_inference,
        "get_period_ai_scores",
        lambda predictor, period_features, models, calibrators, cache=None: {"1h": 0.0, "4h": 0.0, "d1": 0.0},
    )
    monkeypatch.setattr(
        core.ai_inference,
        "get_reg_predictions",
        lambda predictor, period_features, models: (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(core.dynamic_thresholds, "calc_dynamic_threshold", lambda inp: (0.0, 0.0))
    monkeypatch.setattr(
        core.risk_filters, "compute_risk_multipliers", lambda *a, **k: (1.0, 1.0, [], {})
    )
    monkeypatch.setattr(core.position_sizing, "calc_position_size", lambda *a, **k: 0.0)

    captured = {}

    def fake_fuse(combined, ic_weights, strong_confirm_4h, *, cycle_weight=None, conflict_mult=0.7, ic_stats=None, min_agree=2):
        captured["cycle_weight"] = cycle_weight
        captured["conflict_mult"] = conflict_mult
        captured["min_agree"] = min_agree
        return 0.0, False, False, False

    monkeypatch.setattr(core.multi_period_fusion, "fuse_scores", fake_fuse)

    core.generate_signal({}, {}, {})

    assert captured["min_agree"] == 3
    assert captured["conflict_mult"] == 0.5
    assert captured["cycle_weight"] == cfg["fusion"]["cycle_weight"]


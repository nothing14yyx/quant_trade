from quant_trade.signal.model_signal import ModelSignalCfg, model_score_from_proba


def test_symbol_specific_thresholds_and_fallback():
    cfg = ModelSignalCfg(
        symbol_thresholds={"IF": {"p_min_up": 0.6, "p_min_down": 0.6, "margin_min": 0.5}}
    )
    proba = [0.2, 0.1, 0.7]
    assert model_score_from_proba(proba, cfg, symbol="IF") is None
    assert model_score_from_proba(proba, cfg, symbol="OTHER") == 1.0


def test_symbol_specific_down_threshold():
    cfg = ModelSignalCfg(
        symbol_thresholds={"IF": {"p_min_up": 0.6, "p_min_down": 0.7, "margin_min": 0.1}}
    )
    proba = [0.65, 0.1, 0.1]
    assert model_score_from_proba(proba, cfg, symbol="IF") is None
    assert model_score_from_proba(proba, cfg, symbol="OTHER") == -1.0

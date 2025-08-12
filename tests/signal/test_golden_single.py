import json
from pathlib import Path

import pytest

from quant_trade.robust_signal_generator import RobustSignalGenerator, RobustSignalGeneratorConfig
import quant_trade.signal.core as core

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(params=sorted(FIXTURE_DIR.glob("golden*.json")))
def case_data(request):
    return load_json(request.param)

def test_golden_single(case_data, monkeypatch):
    class DummyAI:
        def __init__(self, model_paths):
            self.models = {p: {"up": {"features": []}, "down": {"features": []}} for p in ("1h", "4h", "d1")}
            self.calibrators = {p: {"up": None, "down": None} for p in ("1h", "4h", "d1")}
        def get_ai_score(self, *a, **k):
            return 0.123456

    monkeypatch.setattr(core, "AIModelPredictor", DummyAI)

    cfg = RobustSignalGeneratorConfig(
        model_paths={p: {"up": "dummy", "down": "dummy"} for p in ("1h", "4h", "d1")},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
    )
    rsg = RobustSignalGenerator(cfg)

    features = case_data["features"]
    raw = case_data["raw"]
    result = rsg.generate_signal(
        features["1h"],
        features["4h"],
        features["d1"],
        features.get("15m"),
        raw_features_1h=raw["1h"],
        raw_features_4h=raw["4h"],
        raw_features_d1=raw["d1"],
        raw_features_15m=raw.get("15m"),
        global_metrics=case_data.get("global_metrics"),
        open_interest=case_data.get("open_interest"),
        order_book_imbalance=case_data.get("order_book_imbalance"),
        symbol=case_data.get("symbol"),
    )

    expected = case_data["expected"]

    assert isinstance(result, dict)
    for key in ("signal", "score", "position_size", "details"):
        assert key in result

    assert abs(result["signal"] - expected["signal"]) <= 1e-6
    assert abs(result["score"] - expected["score"]) <= 1e-6
    assert abs(result["position_size"] - expected["position_size"]) <= 1e-6

    details = result["details"]
    for field in ("vote", "protect", "env", "exit", "factors"):
        assert field in details

    rsg.stop_weight_update_thread()

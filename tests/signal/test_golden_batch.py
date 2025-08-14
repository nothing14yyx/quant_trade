import json
from pathlib import Path

import pytest

from quant_trade.robust_signal_generator import RobustSignalGenerator, RobustSignalGeneratorConfig
import quant_trade.signal.core as core

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def cases_data():
    return [load_json(p) for p in sorted(FIXTURE_DIR.glob("golden*.json"))]


def test_golden_batch(cases_data, monkeypatch):
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

    feats_1h = [c["features"]["1h"] for c in cases_data]
    feats_4h = [c["features"]["4h"] for c in cases_data]
    feats_d1 = [c["features"]["d1"] for c in cases_data]
    feats_15m = [c["features"].get("15m") for c in cases_data]
    gm_list = [c.get("global_metrics") for c in cases_data]
    oi_list = [c.get("open_interest") for c in cases_data]
    ob_list = [c.get("order_book_imbalance") for c in cases_data]
    symbols = [c.get("symbol") for c in cases_data]

    results = rsg.generate_signal_batch(
        feats_1h,
        feats_4h,
        feats_d1,
        feats_15m,
        global_metrics=gm_list,
        open_interest=oi_list,
        order_book_imbalance=ob_list,
        symbols=symbols,
    )

    assert len(results) == len(cases_data)
    for res, case in zip(results, cases_data):
        expected = case["expected"]

        assert isinstance(res, dict)
        for key in ("signal", "score", "position_size", "details"):
            assert key in res

        assert abs(res["signal"] - expected["signal"]) <= 1e-6
        assert abs(res["score"] - expected["score"]) <= 1e-2
        assert abs(res["position_size"] - expected["position_size"]) <= 1e-6

        details = res["details"]
        for field in ("vote", "protect", "env", "exit", "factors"):
            assert field in details

    rsg.update_weights()

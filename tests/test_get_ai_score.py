import pandas as pd
import joblib
import pytest
from sklearn.dummy import DummyClassifier

from quant_trade.robust_signal_generator import RobustSignalGenerator, RobustSignalGeneratorConfig
from quant_trade.risk_manager import RiskManager


def build_dummy_model(constant, features, path):
    mdl = DummyClassifier(strategy="constant", constant=constant)
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    mdl.fit(X, y)
    joblib.dump({"pipeline": mdl, "features": features}, path)


def test_get_ai_score_auto_features(tmp_path):
    up_path = tmp_path / "up.pkl"
    down_path = tmp_path / "down.pkl"
    feats = ["f1", "f2"]
    build_dummy_model(1, feats, up_path)
    build_dummy_model(0, feats, down_path)

    cfg = RobustSignalGeneratorConfig(
        model_paths={"1h": {"up": str(up_path), "down": str(down_path)}},
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
    )
    rsg = RobustSignalGenerator(cfg)
    rsg.risk_manager = RiskManager()

    df = pd.DataFrame({"f1": [0.5], "f2": [0.1]})
    score = rsg.predictor.get_ai_score(df, rsg.models["1h"]["up"], rsg.models["1h"]["down"])
    assert score == pytest.approx(1.0)
    rsg.update_weights()

import joblib
from sklearn.dummy import DummyClassifier
from quant_trade.ai_model_predictor import AIModelPredictor


def build_dummy_model(path):
    mdl = DummyClassifier(strategy="constant", constant=0)
    mdl.fit([[0], [1]], [0, 1])
    joblib.dump({"pipeline": mdl, "features": ["x"]}, path)


def test_d1_models_ignored(tmp_path):
    p1 = tmp_path / "m1.pkl"
    p4 = tmp_path / "m4.pkl"
    pd1 = tmp_path / "md.pkl"
    build_dummy_model(p1)
    build_dummy_model(p4)
    build_dummy_model(pd1)

    model_paths = {
        "1h": {"up": str(p1)},
        "4h": {"up": str(p4)},
        "d1": {"up": str(pd1)},
    }
    predictor = AIModelPredictor(model_paths)
    assert set(predictor.models) == {"1h", "4h"}
    assert "d1" not in predictor.calibrators


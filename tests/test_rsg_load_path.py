from quant_trade.robust_signal_generator import RobustSignalGenerator


def test_model_path_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_paths = {"1h": {"cls": "models/model_1h_cls.pkl"}}
    rsg = RobustSignalGenerator(
        model_paths=model_paths,
        feature_cols_1h=[],
        feature_cols_4h=[],
        feature_cols_d1=[],
    )
    assert "cls" in rsg.models.get("1h", {})
    assert hasattr(rsg.models["1h"]["cls"]["pipeline"], "predict")


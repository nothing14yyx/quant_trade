import pandas as pd

from quant_trade.signal import VotingModel


def test_voting_model_train_and_predict(tmp_path):
    data = pd.DataFrame(
        {
            "ai_dir": [1, -1, 1, -1],
            "short_mom_dir": [1, -1, 1, -1],
            "vol_breakout_dir": [1, -1, 1, -1],
            "trend_dir": [1, -1, 1, -1],
            "confirm_dir": [1, -1, 1, -1],
            "ob_dir": [0, 0, 1, -1],
            "label": [1, 0, 1, 0],
        }
    )
    model = VotingModel.train(data)
    prob = model.predict_proba([[1, 1, 1, 1, 1, 0]])[0, 1]
    assert 0.0 <= prob <= 1.0

    path = tmp_path / "vm.pkl"
    model.save(path)
    loaded = VotingModel.load(path)
    prob2 = loaded.predict_proba([[1, 1, 1, 1, 1, 0]])[0, 1]
    assert prob == prob2

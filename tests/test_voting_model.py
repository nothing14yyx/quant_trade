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
            "abs_ai_score": [0.1, 0.2, 0.3, 0.4],
            "abs_momentum": [0.5, 0.6, 0.7, 0.8],
            "consensus_all": [1, 1, 1, 1],
            "ic_weight": [0.9, 0.8, 0.7, 0.6],
            "abs_score_minus_base_th": [0.2, 0.1, 0.3, 0.4],
            "confirm_15m": [1, -1, 1, -1],
            "short_mom": [0.2, 0.3, 0.4, 0.5],
            "ob_imb": [0.1, 0.2, 0.3, 0.4],
            "label": [1, 0, 1, 0],
        }
    )
    model = VotingModel.train(data)
    sample = [[1, 1, 1, 1, 1, 0, 0.1, 0.5, 1, 0.9, 0.2, 1, 0.2, 0.1]]
    prob = model.predict_proba(sample)[0, 1]
    assert 0.0 <= prob <= 1.0

    path = tmp_path / "vm.pkl"
    model.save(path)
    loaded = VotingModel.load(path)
    prob2 = loaded.predict_proba(sample)[0, 1]
    assert prob == prob2

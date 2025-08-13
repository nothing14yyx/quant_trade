import numpy as np
import pandas as pd
import pytest

from quant_trade.signal import VotingModel
from quant_trade.robust_signal_generator import (
    RobustSignalGenerator,
    RobustSignalGeneratorConfig,
)
import quant_trade.robust_signal_generator as rsg_mod


def test_voting_model_train_and_predict(tmp_path):
    """训练并验证 VotingModel 的基本行为。

    DataFrame 根据 ``VotingModel.FEATURE_COLS`` 动态构造, 这样当模型
    新增特征时测试仍能自动适配。"""

    cols = VotingModel.FEATURE_COLS
    # 为每个特征构造四行简单数值
    values = {
        c: [0.1, -0.2, 0.3, -0.4] for c in cols
    }
    values["label"] = [1, 0, 1, 0]
    data = pd.DataFrame(values)

    model = VotingModel.train(data)
    sample = [[data[c].iloc[0] for c in cols]]
    prob = model.predict_proba(sample)[0, 1]
    assert 0.0 <= prob <= 1.0

    path = tmp_path / "vm.pkl"
    model.save(path)
    loaded = VotingModel.load(path)
    prob2 = loaded.predict_proba(sample)[0, 1]
    assert prob == prob2


@pytest.mark.parametrize(
    "prob, weak_vote, strong_confirm",
    [
        (0.2, False, True),
        (0.5, True, False),
        (0.9, False, True),
    ],
)
def test_compute_vote_behavior(monkeypatch, prob, weak_vote, strong_confirm):
    """验证 _compute_vote 对不同概率输出的判定。"""

    cfg = RobustSignalGeneratorConfig(prob_margin=0.1, strong_prob_th=0.8)
    rsg = RobustSignalGenerator(cfg)

    class DummyModel:
        feature_cols = VotingModel.FEATURE_COLS

        def predict_proba(self, X):
            return np.array([[1 - prob, prob]])

    monkeypatch.setattr(rsg_mod, "safe_load", lambda path=None: DummyModel())

    res = rsg._compute_vote(
        fused_score=0.0,
        ai_scores={"1h": 0.0},
        short_mom=0.0,
        vol_breakout=0.0,
        factor_scores={"trend": 0.0},
        score_details={},
        confirm_15m=0.0,
        ob_imb=0.0,
        base_th=0.0,
    )

    direction = 1 if prob >= 0.5 else -1
    expected_vote = direction * max(prob, 1 - prob)
    assert res["vote"] == pytest.approx(expected_vote)
    assert res["weak_vote"] is weak_vote
    assert res["strong_confirm"] is strong_confirm

import types

from quant_trade.signal.ai_inference import get_period_ai_scores, get_reg_predictions


class DummyPredictor:
    def get_ai_score(self, *args, **kwargs):
        return 0.1

    def get_ai_score_cls(self, *args, **kwargs):
        return 0.2

    def get_vol_prediction(self, *args, **kwargs):
        return 0.3

    def get_reg_prediction(self, *args, **kwargs):
        return 0.5


def test_basic_outputs():
    predictor = DummyPredictor()
    feats = {"1h": {"f1": 1}}
    models = {
        "1h": {
            "up": {},
            "down": {},
            "vol": {},
            "rise": {},
            "drawdown": {},
        }
    }
    calibrators = {"1h": {"up": None, "down": None}}

    ai_scores = get_period_ai_scores(predictor, feats, models, calibrators)
    assert set(ai_scores) == {"1h"}
    assert ai_scores["1h"] == 0.1

    vol, rise, draw = get_reg_predictions(predictor, feats, models)
    assert vol["1h"] == 0.3
    assert rise["1h"] == 0.5
    assert draw["1h"] == 0.5

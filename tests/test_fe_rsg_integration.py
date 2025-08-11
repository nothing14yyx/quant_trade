import pandas as pd

from quant_trade.utils.helper import calc_features_raw
from quant_trade.feature_engineering import calc_cross_features
from quant_trade.tests.test_utils import make_dummy_rsg


def test_feature_engineer_to_rsg_signal(monkeypatch):
    times = pd.date_range('2024-01-01', periods=5, freq='h')
    df = pd.DataFrame({
        'open': range(1, 6),
        'high': range(1, 6),
        'low': range(1, 6),
        'close': range(1, 6),
        'volume': 1,
        'social_sentiment': [0, 1, 0, -1, 1],
        'AdrActCnt': range(5),
        'AdrNewCnt': range(5),
        'TxCnt': range(5),
        'CapMrktCurUSD': [100]*5,
        'CapRealUSD': [80]*5,
    }, index=times)

    f1h = calc_features_raw(df, '1h')
    agg = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'social_sentiment': 'mean',
        'AdrActCnt': 'mean', 'AdrNewCnt': 'mean', 'TxCnt': 'mean',
        'CapMrktCurUSD': 'last', 'CapRealUSD': 'last',
    }
    f4h = calc_features_raw(df.resample('4h').agg(agg), '4h')
    fd1 = calc_features_raw(df.resample('1d').agg(agg), 'd1')
    merged = calc_cross_features(f1h, f4h, fd1)

    feat1h = f1h.iloc[-1].to_dict()
    feat4h = f4h.iloc[-1].to_dict()
    featd1 = fd1.iloc[-1].to_dict()

    rsg = make_dummy_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.predictor.get_ai_score = lambda f, u, d: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, w=None: 0.2
    rsg.dynamic_threshold = lambda *a, **k: (0.1, 0.0)
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg._precheck_and_direction = lambda *a, **k: ({'signal': 1, 'score': 0.2, 'position_size': 0.1, 'take_profit': None, 'stop_loss': None, 'details': {}}, 0.2, 1)
    rsg.models = {'1h': {'up': None, 'down': None}, '4h': {'up': None, 'down': None}, 'd1': {'up': None, 'down': None}}

    res = rsg.generate_signal(
        feat1h,
        feat4h,
        featd1,
        raw_features_1h=feat1h,
        raw_features_4h=feat4h,
        raw_features_d1=featd1,
        symbol='BTCUSDT'
    )
    assert isinstance(res, dict)
    assert res['signal'] == 1

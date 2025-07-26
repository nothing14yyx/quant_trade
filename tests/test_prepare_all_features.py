import pandas as pd
import numpy as np
import quant_trade.feature_loader as fl


def test_prepare_all_features_eth_corr(monkeypatch):
    times = pd.date_range('2020-01-01', periods=40, freq='h')
    df = pd.DataFrame({
        'open_time': times,
        'open': np.arange(1, 41),
        'high': np.arange(1, 41) + 0.5,
        'low': np.arange(1, 41) - 0.5,
        'close': np.arange(1, 41) + 0.2,
        'volume': 1,
        'taker_buy_base': 0.5,
        'taker_buy_quote': 0.5,
        'funding_rate': 0.01,
        'cg_price': 1,
        'cg_market_cap': 1,
        'cg_total_volume': 1,
        'btc_close': np.arange(1, 41) + 0.1,
        'eth_close': np.arange(1, 41) + 0.2,
    })

    monkeypatch.setattr(fl, 'load_latest_klines', lambda *a, **k: df)
    scaled1h, scaled4h, scaledd1, raw1h, raw4h, rawd1 = fl.prepare_all_features(None, 'XRPUSDT', {}, 0)
    assert 'eth_correlation_1h_1h' in raw1h
    assert 'eth_correlation_1h_4h' in raw1h

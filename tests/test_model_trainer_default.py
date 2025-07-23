import sys
import importlib
import pandas as pd
import pytest


def test_lambda_default_from_config(monkeypatch):
    monkeypatch.setattr('sqlalchemy.create_engine', lambda *a, **k: None)
    monkeypatch.setattr(pd, 'read_sql', lambda *a, **k: pd.DataFrame(columns=['open_time', 'symbol']))
    sys.modules.pop('quant_trade.model_trainer', None)
    mt = importlib.import_module('quant_trade.model_trainer')
    space = mt.cfg['param_space']['1h']
    assert space['lambda_l2'] == [0.5, 2.0]
    assert mt.get_lambda_default(space, {}) == pytest.approx(1.25)

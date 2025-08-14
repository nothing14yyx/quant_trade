import json
import sys
import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve


def test_compute_metrics_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr('sqlalchemy.create_engine', lambda *a, **k: None)
    monkeypatch.setattr(pd, 'read_sql', lambda *a, **k: pd.DataFrame(columns=['open_time', 'symbol']))
    sys.modules.pop('quant_trade.model_trainer', None)
    mt = importlib.import_module('quant_trade.model_trainer')

    y_true = np.array([0, 1, 2, 1])
    proba = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
        [0.3, 0.4, 0.3],
    ])
    metrics, calib = mt.compute_classification_metrics(y_true, proba)
    assert metrics['LogLoss'] == pytest.approx(log_loss(y_true, proba))
    assert metrics['AUC'] == pytest.approx(
        roc_auc_score(y_true, proba, multi_class='ovr', average='macro')
    )
    y_onehot = np.eye(proba.shape[1])[y_true]
    expected_brier = np.mean(np.sum((proba - y_onehot) ** 2, axis=1))
    assert metrics['Brier'] == pytest.approx(expected_brier)
    for i in range(proba.shape[1]):
        pt, pp = calibration_curve((y_true == i).astype(int), proba[:, i], n_bins=10)
        assert calib[str(i)]['prob_true'] == pt.tolist()
        assert calib[str(i)]['prob_pred'] == pp.tolist()

    report = {
        'splits': {**mt.cv_cfg, 'n_splits': 5},
        'embargo': mt.EMBARGO,
        'metrics': metrics,
        'calibration_curve': calib,
    }
    path = tmp_path / 'report.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f)
    with open(path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert 'AUC' in loaded['metrics']
    assert 'Brier' in loaded['metrics']
    assert len(loaded['calibration_curve']) == proba.shape[1]
    for v in loaded['calibration_curve'].values():
        assert len(v['prob_true']) == len(v['prob_pred'])

import json
import sys
import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


def test_compute_metrics_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr('sqlalchemy.create_engine', lambda *a, **k: None)
    monkeypatch.setattr(pd, 'read_sql', lambda *a, **k: pd.DataFrame(columns=['open_time', 'symbol']))
    sys.modules.pop('quant_trade.model_trainer', None)
    mt = importlib.import_module('quant_trade.model_trainer')

    y_true = np.array([0, 1, 1, 0])
    proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.6, 0.4],
    ])
    metrics, calib = mt.compute_classification_metrics(y_true, proba)
    assert metrics['LogLoss'] == pytest.approx(log_loss(y_true, proba))
    assert metrics['AUC'] == pytest.approx(roc_auc_score(y_true, proba[:, 1]))
    assert metrics['Brier'] == pytest.approx(brier_score_loss(y_true, proba[:, 1]))
    pt, pp = calibration_curve(y_true, proba[:, 1], n_bins=10)
    assert calib['prob_true'] == pt.tolist()
    assert calib['prob_pred'] == pp.tolist()

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
    assert len(loaded['calibration_curve']['prob_true']) == len(
        loaded['calibration_curve']['prob_pred']
    )

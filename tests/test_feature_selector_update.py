import importlib
import pandas as pd
import yaml
import pytest


def test_update_selected_features(tmp_path):
    df = pd.DataFrame({
        'open_time': pd.date_range('2020-01-01', periods=100, freq='h'),
        'f1': range(100),
        'f2': [0.5] * 100,
        'target': [i + 0.1 for i in range(100)],
    })

    fs = importlib.import_module('quant_trade.feature_selector')
    importlib.reload(fs)

    yaml_path = tmp_path / 'sel.yaml'
    yaml_path.write_text(yaml.dump({'1h': ['f1', 'f2']}))

    keep = fs.update_selected_features(
        df, '1h', 'target', yaml_file=yaml_path, shap_thresh=0.05, ic_thresh=0.05
    )

    out = yaml.safe_load(yaml_path.read_text())
    assert out['1h'] == keep
    assert 'f2' not in keep

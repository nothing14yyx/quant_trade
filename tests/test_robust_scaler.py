import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
from utils.robust_scaler import (
    compute_robust_z_params,
    save_scaler_params_to_json,
    load_scaler_params_from_json,
    apply_robust_z_with_params,
)


def test_robust_scaler_roundtrip(tmp_path):
    df = pd.DataFrame({'a': [0, 1, 2, 100]})
    params = compute_robust_z_params(df, ['a'])
    path = tmp_path / 'params.json'
    save_scaler_params_to_json(params, path)
    loaded = load_scaler_params_from_json(path)
    scaled = apply_robust_z_with_params(df, loaded)

    lower, upper = np.percentile(df['a'], [0.5, 99.5])
    clipped = np.clip(df['a'], lower, upper)
    mu = clipped.mean()
    sigma = clipped.to_numpy().std() + 1e-6
    expected = (clipped - mu) / sigma
    assert np.allclose(scaled['a'].to_numpy(), expected)

"""Tests for model signal calibration utilities."""

from __future__ import annotations

import numpy as np

from quant_trade.signal.model_signal import ModelSignalCfg, fit_center_scale


def test_fit_center_scale_recovers_params():
    proba = np.linspace(0.0, 1.0, 101)
    future = np.tanh((proba - 0.5) * 3.0)
    res = fit_center_scale(proba, future)
    assert res["center"] == 0.5
    assert res["scale"] == 3.0


def test_model_signal_cfg_from_calibration_dict():
    calib = {"center": 0.4, "scale": 2.0}
    cfg = ModelSignalCfg.from_calibration(calib)
    assert cfg.center == 0.4
    assert cfg.scale == 2.0

import json
import numpy as np

from quant_trade.run_scheduler import _to_builtin, safe_json_dumps


def test_to_builtin_conversion():
    data = {"flag": np.bool_(True), "num": np.int64(2), "flt": np.float64(1.5)}
    converted = {k: _to_builtin(v) for k, v in data.items()}
    assert converted == {"flag": 1, "num": 2, "flt": 1.5}
    # ensure json dumps succeeds
    json.dumps(converted)


def test_json_dumps_default():
    data = {"flag": np.bool_(False), "num": np.int64(5), "flt": np.float64(0.7)}
    dumped = safe_json_dumps(data)
    assert dumped == json.dumps({"flag": 0, "num": 5, "flt": 0.7})


def test_json_nan_conversion():
    dumped = safe_json_dumps({"val": np.nan})
    assert dumped == "{\"val\": null}"

def test_dict_nan_conversion():
    data = {"val": np.nan, "num": np.float64(1.2)}
    converted = {k: _to_builtin(v) for k, v in data.items()}
    assert converted == {"val": None, "num": 1.2}


def test_to_builtin_none_returns_none():
    assert _to_builtin(None) is None

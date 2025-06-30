import numpy as np
import pytest

from quant_trade.tests.test_utils import make_dummy_rsg


def test_combine_score_vectorized_matches_loop():
    rsg = make_dummy_rsg()
    ai = np.array([0.1, 0.2, -0.1])
    fs = {
        'trend': np.array([0.0, 0.1, 0.2]),
        'momentum': np.array([0.2, 0.2, 0.2]),
        'volatility': np.array([0.1, 0.1, 0.1]),
        'volume': np.array([0.0, 0.0, 0.0]),
        'sentiment': np.array([0.05, 0.05, 0.05]),
        'funding': np.array([0.0, 0.0, 0.0]),
    }

    res_vec = rsg.combine_score_vectorized(ai, fs, rsg.base_weights)
    expected = np.array([
        rsg.combine_score(ai[i], {k: v[i] for k, v in fs.items()}, rsg.base_weights)
        for i in range(ai.size)
    ])
    assert np.allclose(res_vec, expected)


def test_calc_factor_scores_vectorized_matches_loop():
    rsg = make_dummy_rsg()
    ai = {
        '1h': np.array([0.1, 0.2]),
        '4h': np.array([-0.1, 0.1]),
        'd1': np.array([0.0, 0.2]),
    }
    fs = {p: {
        'trend': np.array([0.1, 0.1]),
        'momentum': np.array([0.1, 0.1]),
        'volatility': np.array([0.1, 0.1]),
        'volume': np.array([0.1, 0.1]),
        'sentiment': np.array([0.1, 0.1]),
        'funding': np.array([0.1, 0.1]),
    } for p in ('1h', '4h', 'd1')}

    vec = rsg.calc_factor_scores_vectorized(ai, fs, rsg.base_weights)

    expected = {}
    for p in ('1h', '4h', 'd1'):
        w = rsg.base_weights.copy()
        if p in ('1h', '4h'):
            for k in ('trend', 'momentum', 'volume'):
                w[k] *= 0.7
        arr = []
        for i in range(ai[p].size):
            fs_i = {k: v[i] for k, v in fs[p].items()}
            arr.append(rsg.combine_score(ai[p][i], fs_i, w))
        expected[p] = np.array(arr)

    for p in expected:
        assert np.allclose(vec[p], expected[p])

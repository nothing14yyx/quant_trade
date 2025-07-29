import pytest
import math
from quant_trade.tests.test_utils import make_dummy_rsg



def test_apply_oi_overheat_protection():
    rsg = make_dummy_rsg()
    rsg.oi_scale = 0.7
    val, flag = rsg.apply_oi_overheat_protection(1.0, 0.1, 0.3)
    assert flag is False
    assert val == pytest.approx(1.0 * (1 + 0.03 * 0.1))
    val2, flag2 = rsg.apply_oi_overheat_protection(1.0, 0.4, 0.3)
    assert flag2 is True
    assert val2 == pytest.approx(1.0 * 0.7)


def test_detect_reversal_adjusts_threshold():
    rsg = make_dummy_rsg()
    rsg.risk_score_limit = 5.0
    rsg.detect_reversal = lambda *a, **k: 1

    def dyn_th(atr, adx, funding=0, **kwargs):
        if kwargs.get('reversal'):
            return 0.2, 0.1
        return 0.1, 0.1

    rsg.dynamic_threshold = dyn_th
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.get_ai_score = lambda *a, **k: 0
    rsg.get_factor_scores = lambda f, p: {k: 0 for k in rsg.base_weights if k != 'ai'}
    rsg.combine_score = lambda ai, fs, w=None: ai
    rsg.compute_tp_sl = lambda *a, **k: (0, 0)
    rsg.models = {'1h': {'up': None, 'down': None},
                  '4h': {'up': None, 'down': None},
                  'd1': {'up': None, 'down': None}}

    feats_1h = {'close': 100, 'atr_pct_1h': 0.05, 'adx_1h': 0,
                'funding_rate_1h': 0, 'vol_ma_ratio_1h': 1.0}
    feats_4h = {'atr_pct_4h': 0}
    feats_d1 = {}

    res = rsg.generate_signal(feats_1h, feats_4h, feats_d1,
                              raw_features_1h=feats_1h)
    assert res['details']['exit']['dynamic_th_final'] == pytest.approx(0.2)


def test_crowding_factor_reduces_position():
    rsg = make_dummy_rsg()
    cfg = {'min_pos': 0.0}
    params = dict(
        grad_dir=1.0,
        base_coeff=0.4,
        confidence_factor=1.0,
        vol_ratio=1.0,
        fused_score=0.11,
        base_th=0.1,
        regime='trend',
        vol_p=None,
        atr=0.0,
        risk_score=0.3,
        cfg_th_sig=cfg,
        direction=1,
        exit_mult=1.0,
        consensus_all=False,
    )
    full, _, _, _ = rsg.compute_position_size(crowding_factor=1.0, **params)
    reduced, _, _, _ = rsg.compute_position_size(crowding_factor=0.5, **params)
    params_no_risk = params.copy()
    params_no_risk["risk_score"] = 0.0
    base_full, _, _, _ = rsg.compute_position_size(crowding_factor=1.0, **params_no_risk)
    assert reduced == pytest.approx(full * 0.5)
    assert full == pytest.approx(base_full * math.exp(-rsg.risk_scale * 0.3))

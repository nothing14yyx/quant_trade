import pytest
import numpy as np
from collections import deque, OrderedDict
from quant_trade.robust_signal_generator import RobustSignalGenerator
from quant_trade.signal.predictor_adapter import PredictorAdapter
from quant_trade.signal.factor_scorer import FactorScorerImpl
from quant_trade.signal.fusion_rule import FusionRuleBased
from quant_trade.signal.risk_filters import RiskFiltersImpl
from quant_trade.signal.position_sizer import PositionSizerImpl


def make_simple_rsg():
    rsg = RobustSignalGenerator.__new__(RobustSignalGenerator)
    rsg._factor_cache = OrderedDict()
    rsg.factor_scorer = FactorScorerImpl(rsg)
    rsg.history_scores = deque(maxlen=10)
    rsg.oi_change_history = deque(maxlen=10)
    rsg.base_weights = {'ai':1,'trend':1,'momentum':1,'volatility':1,'volume':1,'sentiment':1,'funding':1}
    rsg.current_weights = rsg.base_weights.copy()
    rsg.vote_params = {'weight_ai':2.0,'strong_min':5,'conf_min':1.0}
    rsg.min_weight_ratio = 0.2
    rsg._prev_raw = {p: None for p in ("1h","4h","d1")}
    rsg.sentiment_alpha = 0.5
    rsg.cap_positive_scale = 0.4
    rsg.volume_guard_params = {
        'weak': 0.7,
        'over': 0.9,
        'ratio_low': 0.8,
        'ratio_high': 2.0,
        'roc_low': -20,
        'roc_high': 100,
    }
    rsg.ob_th_params = {'min_ob_th': 0.10, 'dynamic_factor': 0.08}
    rsg.regime_adx_trend = 25
    rsg.regime_adx_range = 20
    rsg.risk_adjust_factor = 0.9
    rsg.risk_adjust_threshold = 0
    rsg.risk_score_limit = 2.0
    rsg.crowding_limit = 1.1
    rsg.max_position = 0.3
    rsg.risk_scale = 1.0
    rsg.min_pos_vol_scale = 0.0
    rsg.volume_quantile_low = 0.2
    rsg.volume_quantile_high = 0.8
    rsg.volume_ratio_history = deque([0.8, 1.0, 1.2], maxlen=10)
    rsg.flip_confirm_bars = 3
    rsg.predictor = PredictorAdapter(None)
    rsg.fusion_rule = FusionRuleBased(rsg)
    rsg.consensus_check = rsg.fusion_rule.consensus_check
    rsg.crowding_protection = rsg.fusion_rule.crowding_protection
    rsg.fuse = rsg.fusion_rule.fuse
    rsg.fuse_multi_cycle = rsg.fusion_rule.fuse
    rsg.risk_filters = RiskFiltersImpl(rsg)
    rsg.position_sizer = PositionSizerImpl(rsg)
    return rsg


def test_layer_scores_product():
    rsg = make_simple_rsg()
    rsg.dynamic_weight_update = lambda: rsg.base_weights
    rsg.predictor.get_ai_score = lambda f,u,d: 0.5
    rsg.factor_scorer.score = lambda f,p:{k:0 for k in rsg.base_weights if k!='ai'}
    rsg.combine_score = lambda ai,fs,w=None: ai
    rsg.dynamic_threshold = lambda *a,**k: (0, 0)
    rsg.position_sizer.compute_tp_sl = lambda *a,**k:(0,0)
    rsg.models={'1h':{'up':None,'down':None},'4h':{'up':None,'down':None},'d1':{'up':None,'down':None}}

    feats={'close':100,'atr_pct_1h':0,'adx_1h':0,'funding_rate_1h':0}
    res = rsg.generate_signal(feats, {'atr_pct_4h':0}, {}, symbol='BTC')
    env = res['details']['env']
    raw = env['logic_score'] * env['env_score']
    expected = np.tanh(raw * (1 - 0.9))
    assert res['score'] == pytest.approx(expected)

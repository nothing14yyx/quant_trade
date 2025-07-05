# -*- coding: utf-8 -*-
import numpy as np
import math
from quant_trade.utils.soft_clip import soft_clip

import pandas as pd
from collections import Counter, deque, OrderedDict
from pathlib import Path
import yaml
import json
import threading
import logging
import time

from .config_manager import ConfigManager
from .ai_model_predictor import AIModelPredictor
from .risk_manager import RiskManager
from .feature_processor import FeatureProcessor

logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)
# Set module logger to WARNING by default so importing modules can
# configure their own verbosity without receiving this module's INFO logs.
logger.setLevel(logging.WARNING)
# 默认配置路径
CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


# 退出信号滞后 bar 数默认值
EXIT_LAG_BARS_DEFAULT = 0

# AI 投票与仓位参数默认值


def _load_default_ai_dir_eps(path: Path) -> float:
    """从配置文件读取 ai_dir_eps，若失败则返回 0.04"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("vote_system", {}).get("ai_dir_eps", 0.04)
    except Exception:
        return 0.04


DEFAULT_AI_DIR_EPS = _load_default_ai_dir_eps(CONFIG_PATH)
DEFAULT_POS_K_RANGE = 0.40    # 震荡市仓位乘数
DEFAULT_POS_K_TREND = 0.60    # 趋势市仓位乘数
DEFAULT_LOW_BASE = 0.06       # 动态阈值下限
DEFAULT_LOW_VOL_RATIO = 0.2   # 低量能阈值

from dataclasses import dataclass
from quant_trade.utils import get_cfg_value, collect_feature_cols


@dataclass
class SignalThresholdParams:
    """Container for all signal threshold related parameters."""

    base_th: float = 0.08
    gamma: float = 0.9
    min_pos: float = 0.05
    quantile: float = 0.80
    low_base: float = DEFAULT_LOW_BASE
    rev_boost: float = 0.15
    rev_th_mult: float = 0.60
    atr_mult: float = 4.0
    funding_mult: float = 8.0
    adx_div: float = 100.0

    @classmethod
    def from_cfg(cls, cfg: dict | None):
        cfg = cfg or {}
        return cls(
            base_th=float(get_cfg_value(cfg, "base_th", cls.base_th)),
            gamma=float(get_cfg_value(cfg, "gamma", cls.gamma)),
            min_pos=float(get_cfg_value(cfg, "min_pos", cls.min_pos)),
            quantile=float(get_cfg_value(cfg, "quantile", cls.quantile)),
            low_base=float(get_cfg_value(cfg, "low_base", cls.low_base)),
            rev_boost=float(get_cfg_value(cfg, "rev_boost", cls.rev_boost)),
            rev_th_mult=float(get_cfg_value(cfg, "rev_th_mult", cls.rev_th_mult)),
            atr_mult=float(get_cfg_value(cfg, "atr_mult", cls.atr_mult)),
            funding_mult=float(get_cfg_value(cfg, "funding_mult", cls.funding_mult)),
            adx_div=float(get_cfg_value(cfg, "adx_div", cls.adx_div)),
        )


@dataclass
class RobustSignalGeneratorConfig:
    """初始化 ``RobustSignalGenerator`` 所需的参数容器."""

    model_paths: dict
    feature_cols_1h: list[str]
    feature_cols_4h: list[str]
    feature_cols_d1: list[str]
    history_window: int = 532
    symbol_categories: dict[str, str] | None = None
    config_path: str | Path = CONFIG_PATH
    core_keys: dict | None = None
    delta_params: dict | None = None
    min_weight_ratio: float = 0.6
    th_window: int = 60
    th_decay: float = 2.0

    @classmethod
    def from_file(cls, path: str | Path):
        """从 YAML/JSON 文件加载配置."""
        with open(path, "r", encoding="utf-8") as f:
            if str(path).endswith((".yml", ".yaml")):
                cfg = yaml.safe_load(f) or {}
            else:
                cfg = json.load(f)
        return cls.from_cfg(cfg, path)

    @classmethod
    def from_cfg(cls, cfg: dict, path: str | Path | None = None):
        """从字典创建配置对象."""
        path = path or cfg.get("config_path", CONFIG_PATH)
        db_cfg = cfg.get("delta_boost", {})
        return cls(
            model_paths=cfg.get("models", {}),
            feature_cols_1h=collect_feature_cols(cfg, "1h"),
            feature_cols_4h=collect_feature_cols(cfg, "4h"),
            feature_cols_d1=collect_feature_cols(cfg, "d1"),
            history_window=cfg.get("history_window", 532),
            symbol_categories=cfg.get("symbol_categories"),
            config_path=path,
            core_keys=db_cfg.get("core_keys"),
            delta_params=db_cfg.get("params"),
            min_weight_ratio=cfg.get("min_weight_ratio", 0.6),
            th_window=cfg.get("th_window", 60),
            th_decay=cfg.get("th_decay", 2.0),
        )


def robust_signal_generator(model, *args, **kwargs):
    """Safely call ``model.generate_signal`` and catch common errors."""
    try:
        return model.generate_signal(*args, **kwargs)
    except (ValueError, KeyError, TypeError) as e:
        logger.info("generate_signal failed: %s", e)
        return None


def softmax(x):
    """简单 softmax 实现"""
    arr = np.array(x, dtype=float)
    ex = np.exp(arr - np.nanmax(arr))
    return ex / ex.sum()


def sigmoid(x):
    """标准 sigmoid 函数"""
    return 1 / (1 + np.exp(-x))


def weighted_quantile(values, q, sample_weight=None):
    """Return the weighted quantile of *values* at quantile *q*."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if sample_weight is None:
        return float(np.quantile(values, q))
    sample_weight = np.asarray(sample_weight, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, values))


@dataclass
class DynamicThresholdInput:
    """Container for metrics used in dynamic threshold calculation."""

    atr: float
    adx: float
    funding: float = 0.0
    atr_4h: float | None = None
    adx_4h: float | None = None
    atr_d1: float | None = None
    adx_d1: float | None = None
    pred_vol: float | None = None
    pred_vol_4h: float | None = None
    pred_vol_d1: float | None = None
    vix_proxy: float | None = None
    regime: str | None = None
    reversal: bool = False


def _calc_history_base(history, base, quantile, window, decay, limit=None):
    """Helper to compute threshold base from history with optional decay."""
    if not history:
        return base
    arr = np.asarray(list(history)[-window:], dtype=float)
    arr = np.abs(arr)
    if arr.size == 0:
        return base
    if decay and decay != 1.0:
        w = np.exp(-decay * np.arange(arr.size)[::-1])
        qv = weighted_quantile(arr, quantile, w)
    else:
        qv = float(np.quantile(arr, quantile))
    if not math.isnan(qv):
        base = max(base, qv)
    if limit is not None and base > limit:
        base = limit
    return base


def adjust_score(
    score: float,
    sentiment: float,
    alpha: float = 0.5,
    *,
    cap_scale: float = 0.7,
    cap_threshold: float = -0.5,
) -> float:
    """根据情绪值调整分数并在负面过强时进一步削弱"""

    if abs(sentiment) <= 0.5:
        return score

    scale = 1 + alpha * np.sign(score) * sentiment
    scale = float(np.clip(scale, 0.6, 1.5))
    adjusted = score * scale

    return cap_positive(adjusted, sentiment, cap_scale, cap_threshold)


def volume_guard(
    score: float,
    ratio: float | None,
    roc: float | None,
    *,
    weak: float = 0.85,
    over: float = 0.9,
    ratio_low: float = -0.5,
    ratio_high: float = 1.5,
    roc_low: float = -20,
    roc_high: float = 100,
) -> float:
    """量能不足或异常时压缩得分"""
    if ratio is None or roc is None:
        return score
    if ratio < ratio_low or roc < roc_low:
        return score * weak
    extreme_ratio = ratio_high * 2
    extreme_roc = roc_high * 1.5
    if ratio_high <= ratio < extreme_ratio and roc_low < roc < extreme_roc:
        mult = 1 + 0.05 * np.sign(score)
        return score * mult
    if ratio >= extreme_ratio or roc >= extreme_roc:
        mult = 1 + (over - 1) * np.sign(score)
        return score * mult
    return score


def cap_positive(
    score: float,
    sentiment: float,
    scale: float = 0.7,
    threshold: float = -0.5,
) -> float:
    """若负面情绪过强则按比例削弱正分"""
    if sentiment <= threshold and score > 0:
        return score * scale
    if sentiment >= 0.5 and score < 0:
        return score * scale
    return score


def fused_to_risk(
    fused_score: float,
    logic_score: float,
    env_score: float,
    *,
    cap: float = 5.0,
) -> float:
    """按安全分母计算并限制 risk score"""
    return RiskManager(cap).fused_to_risk(fused_score, logic_score, env_score)




def sigmoid_dir(score: float, base_th: float, gamma: float) -> float:
    """根据分数计算梯度方向强度, 结果范围 [-1, 1]"""
    amp = np.tanh((abs(score) - base_th) / gamma)
    return np.sign(score) * max(0.0, amp)


def sigmoid_confidence(vote: float, strong_min: float, conf_min: float = 1.0) -> float:
    """根据投票结果计算置信度, 默认下限为1"""
    conf = 1 / (1 + np.exp(-4 * (abs(vote) - strong_min)))
    return max(conf_min, conf)

class RobustSignalGenerator:
    """多周期 AI + 多因子 融合信号生成器。

    - 支持动态阈值与极端行情防护
    - 可通过 :func:`update_ic_scores` 读取历史数据并计算因子 IC
      用于动态调整权重
    - Δ-boost 逻辑现已参数化，可通过 config 或 core_keys / delta_params 定制。
    """

    DEFAULT_CORE_KEYS = {
        "1h": [
            "rsi_1h",
            "macd_hist_1h",
            "ema_diff_1h",
            "atr_pct_1h",
            "vol_ma_ratio_long_1h",
            "funding_rate_1h",
            "support_level_1h",
            "resistance_level_1h",
            "break_support_1h",
            "break_resistance_1h",
            "pivot_r1_1h",
            "pivot_s1_1h",
            "close_vs_pivot_1h",
            "close_vs_vpoc_1h",
        ],
        "4h": [
            "rsi_4h",
            "macd_hist_4h",
            "ema_diff_4h",
            "support_level_4h",
            "resistance_level_4h",
            "break_support_4h",
            "break_resistance_4h",
            "pivot_r1_4h",
            "pivot_s1_4h",
            "close_vs_pivot_4h",
            "close_vs_vpoc_4h",
        ],
        "d1": [
            "rsi_d1",
            "macd_hist_d1",
            "ema_diff_d1",
            "support_level_d1",
            "resistance_level_d1",
            "break_support_d1",
            "break_resistance_d1",
            "pivot_r1_d1",
            "pivot_s1_d1",
            "close_vs_pivot_d1",
            "close_vs_vpoc_d1",
        ],
        "15m": [
            "rsi_15m",
            "macd_hist_15m",
            "ema_diff_15m",
            "vol_ma_ratio_long_15m",
            "boll_perc_15m",
        ],
    }

    DELTA_PARAMS = {
        "rsi": (5, 1.0, 0.05),
        "macd_hist": (0.002, 100.0, 0.05),
        "ema_diff": (0.001, 100.0, 0.03),
        "atr_pct": (0.002, 100.0, 0.03),
        "vol_ma_ratio": (0.2, 1.0, 0.03),
        "funding_rate": (0.0005, 10000, 0.03),
        "close_vs_pivot": (0.01, 20, 0.04),
        "close_vs_vpoc": (0.01, 20, 0.04),
    }

    VOTE_PARAMS = {
        "weight_ai": 3,
        "strong_min": 3,
        "conf_min": 0.30,
    }

    def __init__(self, config: RobustSignalGeneratorConfig):
        model_paths = config.model_paths
        feature_cols_1h = config.feature_cols_1h
        feature_cols_4h = config.feature_cols_4h
        feature_cols_d1 = config.feature_cols_d1
        history_window = config.history_window
        symbol_categories = config.symbol_categories
        config_path = config.config_path
        core_keys = config.core_keys
        delta_params = config.delta_params
        min_weight_ratio = config.min_weight_ratio
        th_window = config.th_window
        th_decay = config.th_decay

        # 多线程访问历史数据时的互斥锁
        # 使用 RLock 以便在部分函数中嵌套调用
        self._lock = threading.RLock()

        # 使用独立模块加载 AI 模型
        self.ai_predictor = AIModelPredictor(model_paths)
        self.models = self.ai_predictor.models
        self.calibrators = self.ai_predictor.calibrators

        # 特征处理器
        self.feature_processor = FeatureProcessor(
            feature_cols_1h, feature_cols_4h, feature_cols_d1
        )

        # 保留原始特征列属性供外部使用
        self.feature_cols_1h = feature_cols_1h
        self.feature_cols_4h = feature_cols_4h
        self.feature_cols_d1 = feature_cols_d1

        # 缓存标准化特征列索引，减少 DataFrame 查找开销
        self._std_index_cache = {p: None for p in ("15m", "1h", "4h", "d1")}

        # 配置管理
        self.config_manager = ConfigManager(config_path)
        cfg = self.config_manager.cfg
        self.cfg = cfg
        self.signal_threshold_cfg = get_cfg_value(cfg, "signal_threshold", {})
        if "low_base" not in self.signal_threshold_cfg:
            self.signal_threshold_cfg["low_base"] = DEFAULT_LOW_BASE
        db_cfg = get_cfg_value(cfg, "delta_boost", {})
        self.core_keys = core_keys or get_cfg_value(db_cfg, "core_keys", self.DEFAULT_CORE_KEYS)
        self.delta_params = delta_params or get_cfg_value(db_cfg, "params", self.DELTA_PARAMS)
        vote_cfg = get_cfg_value(cfg, "vote_system", {})
        self.vote_params = {
            "weight_ai": vote_cfg.get("weight_ai", self.VOTE_PARAMS["weight_ai"]),
            "strong_min": vote_cfg.get("strong_min", self.VOTE_PARAMS["strong_min"]),
            "conf_min": vote_cfg.get("conf_min", self.VOTE_PARAMS["conf_min"]),
        }
        self.ai_dir_eps = get_cfg_value(vote_cfg, "ai_dir_eps", DEFAULT_AI_DIR_EPS)
        self.vote_weights = get_cfg_value(
            cfg,
            "vote_weights",
            {
                "ai": self.vote_params["weight_ai"],
                "mom": 1,
                "vol_breakout": 1,
                "trend": 1,
                "confirm_15m": 1,
            },
        )

        filters_cfg = get_cfg_value(cfg, "signal_filters", {})
        # ↓ 放宽阈值，防止信号被过度过滤
        self.signal_filters = {
            "min_vote": get_cfg_value(filters_cfg, "min_vote", 2),
            "confidence_vote": get_cfg_value(filters_cfg, "confidence_vote", 0.12),
        }

        pc_cfg = get_cfg_value(cfg, "position_coeff", {})
        self.pos_coeff_range = get_cfg_value(pc_cfg, "range", DEFAULT_POS_K_RANGE)
        self.pos_coeff_trend = get_cfg_value(pc_cfg, "trend", DEFAULT_POS_K_TREND)
        self.low_vol_ratio = get_cfg_value(
            cfg,
            "low_vol_ratio",
            get_cfg_value(pc_cfg, "low_vol_ratio", DEFAULT_LOW_VOL_RATIO),
        )

        self.sentiment_alpha = get_cfg_value(cfg, "sentiment_alpha", 0.5)
        self.cap_positive_scale = get_cfg_value(cfg, "cap_positive_scale", 0.7)
        self.tp_sl_cfg = get_cfg_value(cfg, "tp_sl", {})
        vg_cfg = get_cfg_value(cfg, "volume_guard", {})
        self.volume_guard_params = {
            "weak": get_cfg_value(vg_cfg, "weak_scale", 0.90),
            "over": get_cfg_value(vg_cfg, "over_scale", 0.9),
            "ratio_low": get_cfg_value(vg_cfg, "ratio_low", 0.5),
            "ratio_high": get_cfg_value(vg_cfg, "ratio_high", 2.0),
            "roc_low": get_cfg_value(vg_cfg, "roc_low", -20),
            "roc_high": get_cfg_value(vg_cfg, "roc_high", 100),
        }
        ob_cfg = get_cfg_value(cfg, "ob_threshold", {})
        self.ob_th_params = {
            "min_ob_th": get_cfg_value(ob_cfg, "min_ob_th", 0.15),
            "dynamic_factor": get_cfg_value(ob_cfg, "dynamic_factor", 0.08),
        }
        self.risk_score_cap = get_cfg_value(cfg, "risk_score_cap", 5.0)
        # 风险管理器
        self.risk_manager = RiskManager(cap=self.risk_score_cap)
        self.exit_lag_bars = get_cfg_value(cfg, "exit_lag_bars", EXIT_LAG_BARS_DEFAULT)
        oi_cfg = get_cfg_value(cfg, "oi_protection", {})
        self.oi_scale = get_cfg_value(oi_cfg, "scale", 0.8)
        self.max_same_direction_rate = get_cfg_value(oi_cfg, "crowding_threshold", 0.95)
        self.veto_level = get_cfg_value(cfg, "veto_level", 0.7)
        self.flip_coeff = get_cfg_value(cfg, "flip_coeff", 0.3)
        cw_cfg = get_cfg_value(cfg, "cycle_weight", {})
        self.cycle_weight = {
            "strong": get_cfg_value(cw_cfg, "strong", 1.2),
            "weak": get_cfg_value(cw_cfg, "weak", 0.8),
            "opposite": get_cfg_value(cw_cfg, "opposite", 0.5),
        }
        regime_cfg = get_cfg_value(cfg, "regime", {})
        self.regime_adx_trend = get_cfg_value(regime_cfg, "adx_trend", 25)
        self.regime_adx_range = get_cfg_value(regime_cfg, "adx_range", 20)

        risk_adj_cfg = get_cfg_value(cfg, "risk_adjust", {})
        self.risk_adjust_factor = get_cfg_value(risk_adj_cfg, "factor", 0.9)
        self.risk_adjust_threshold = get_cfg_value(
            cfg, "risk_adjust_threshold", get_cfg_value(risk_adj_cfg, "threshold", 0.01)
        )

        protect_cfg = get_cfg_value(cfg, "protection_limits", {})
        self.risk_score_limit = get_cfg_value(protect_cfg, "risk_score", 2.00)
        self.crowding_limit = get_cfg_value(
            cfg, "crowding_limit", get_cfg_value(protect_cfg, "crowding", 1.05)
        )

        self.max_position = get_cfg_value(cfg, "max_position", 0.3)
        self.min_trend_align = get_cfg_value(cfg, "min_trend_align", 1)
        self.th_down_d1 = get_cfg_value(self.cfg, "th_down_d1", 0.74)
        self.min_weight_ratio = min_weight_ratio
        self.th_window = th_window
        self.th_decay = th_decay

        # 静态因子权重（后续可由动态IC接口进行更新）
        _base_weights = {
            'ai': 0.1974973877121397,
            'trend': 0.12014347730398978,
            'momentum': 0.28250689231946036,
            'volatility': 0.17943635201167124,
            'volume': 0.13887479231887473,
            'sentiment': 0.06500720882489561,
            'funding': 0.1445347730557763,
        }
        total_w = sum(_base_weights.values())
        if total_w <= 0:
            total_w = 1.0
        self.base_weights = {k: v / total_w for k, v in _base_weights.items()}

        # 当前权重，初始与 base_weights 相同
        self.current_weights = self.base_weights.copy()

        # 初始化各因子对应的IC分数，可在配置文件中覆盖
        cfg_ic = cfg.get("ic_scores")
        if isinstance(cfg_ic, dict):
            self.ic_scores = {k: float(cfg_ic.get(k, 1.0)) for k in self.base_weights.keys()}
        else:
            self.ic_scores = {k: 1.0 for k in self.base_weights.keys()}
        # 保存各因子IC的滑窗历史，便于做滚动平均
        self.ic_history = {k: deque(maxlen=history_window) for k in self.base_weights.keys()}

        # 用于存储历史融合得分，方便计算动态阈值（最大长度由 history_window 指定）
        self.history_scores = deque(maxlen=history_window)

        # 全局得分列表用于拥挤度保护
        self.all_scores_list = deque(maxlen=500)

        # 保存近期 OI 变化率，便于自适应过热阈值
        self.oi_change_history = deque(maxlen=history_window)
        # 记录BTC Dominance历史，计算短期与长期差异
        self.btc_dom_history = deque(maxlen=history_window)
        # 记录ETH Dominance历史，供市场偏好判断
        self.eth_dom_history = deque(maxlen=history_window)

        # 币种与板块的映射，用于板块热度修正
        self.symbol_categories = {k.upper(): v for k, v in (symbol_categories or {}).items()}
        # 各币种独立缓存
        self.symbol_data = {}

        # 缓存原始特征与内部状态
        self._raw_history = {
            "15m": deque(maxlen=4),
            "1h": deque(maxlen=4),
            "4h": deque(maxlen=2),
            "d1": deque(maxlen=2),
        }
        self._prev_raw = {p: None for p in ("1h", "4h", "d1")}
        self._last_signal = 0
        self._last_score = 0.0
        self._prev_vote = 0
        self._exit_lag = 0
        self._cooldown = 0
        self._volume_checked = False
        self._equity_drawdown = 0.0

        # 缓存计算结果，避免重复计算
        self.cache_maxsize = 1000
        self._ai_score_cache = OrderedDict()
        self._factor_cache = OrderedDict()
        self._factor_score_cache = OrderedDict()
        self._fuse_cache = OrderedDict()


        # 当多个信号方向过于集中时，用于滤除极端行情（最大同向信号比例阈值）
        # 值由配置 oi_protection.crowding_threshold 控制

        # 定时更新因子权重
        self.start_weight_update_thread()

    def __getattr__(self, name):
        defaults = {
            "history_scores": deque(maxlen=3000),
            "oi_change_history": deque(maxlen=3000),
            "btc_dom_history": deque(maxlen=3000),
            "eth_dom_history": deque(maxlen=3000),
            "ic_history": {k: deque(maxlen=3000) for k in getattr(self, "base_weights", {})},
            "_lock": threading.RLock(),
            "_prev_raw": {p: None for p in ("1h", "4h", "d1")},
            "_raw_history": {
                "15m": deque(maxlen=4),
                "1h": deque(maxlen=4),
                "4h": deque(maxlen=2),
                "d1": deque(maxlen=2),
            },
            "symbol_data": {},
            "calibrators": {p: {"up": None, "down": None} for p in ("1h", "4h", "d1")},
            "core_keys": self.DEFAULT_CORE_KEYS.copy(),
            "delta_params": self.DELTA_PARAMS.copy(),
            "vote_params": self.VOTE_PARAMS.copy(),
            "vote_weights": {"ob": 4, "short_mom": 2, "fast_mom": 0.5, "ai": 3, "vol_breakout": 1},
            "exit_lag_bars": EXIT_LAG_BARS_DEFAULT,
            "th_window": 60,
            "th_decay": 2.0,
            "risk_manager": RiskManager(),
            "all_scores_list": deque(maxlen=500),
            "_equity_drawdown": 0.0,
            "_last_score": 0.0,
            "_last_signal": 0,
            "_prev_vote": 0,
            "_exit_lag": 0,
            "_cooldown": 0,
            "_volume_checked": False,
            "pos_coeff_range": DEFAULT_POS_K_RANGE,
            "pos_coeff_trend": DEFAULT_POS_K_TREND,
            "low_vol_ratio": DEFAULT_LOW_VOL_RATIO,
            "ai_dir_eps": DEFAULT_AI_DIR_EPS,
            "cycle_weight": {"strong": 1.2, "weak": 0.8, "opposite": 0.5},
            "flip_coeff": 0.3,
            "veto_level": 0.7,
            "ic_scores": {},
            "th_down_d1": 0.74,
            "min_trend_align": 1,
            "_ai_score_cache": OrderedDict(),
            "_factor_cache": OrderedDict(),
            "_factor_score_cache": OrderedDict(),
            "_fuse_cache": OrderedDict(),
            "cache_maxsize": 1000,
        }
        if name in defaults:
            val = defaults[name]
            setattr(self, name, val)
            return val
        raise AttributeError(name)

    @property
    def signal_threshold_cfg(self):
        if not hasattr(self, "_signal_threshold_cfg"):
            self._signal_threshold_cfg = {
                "base_th": 0.08,
                "gamma": 0.9,
                "min_pos": 0.10,
                "low_base": DEFAULT_LOW_BASE,
                "rev_boost": 0.15,
                "rev_th_mult": 0.60,
                "atr_mult": 4.0,
                "funding_mult": 8.0,
                "adx_div": 100.0,
            }
            self.signal_params = SignalThresholdParams.from_cfg(self._signal_threshold_cfg)
        return self._signal_threshold_cfg

    @signal_threshold_cfg.setter
    def signal_threshold_cfg(self, value):
        self._signal_threshold_cfg = value or {}
        self.signal_params = SignalThresholdParams.from_cfg(self._signal_threshold_cfg)


    def get_dynamic_oi_threshold(self, pred_vol=None, base=0.5, quantile=0.9):
        """根据历史 OI 变化率及预测波动率自适应阈值"""
        with self._lock:
            history = list(self.oi_change_history)
        if not history:
            base = 0.2
        th = _calc_history_base(
            history,
            base,
            quantile,
            self.th_window,
            self.th_decay,
        )
        if pred_vol is not None:
            th += min(0.1, abs(pred_vol) * 0.5)
        return max(th, 0.30)

    def classify_regime(self, adx, bb_width, channel_pos):
        """根据ADX和布林带宽度变化判别市场状态"""
        if adx is None or bb_width is None:
            return "unknown"
        try:
            adx = float(adx)
            bb_chg = float(bb_width)
        except Exception:
            return "unknown"
        if adx >= self.regime_adx_trend and bb_chg > 0:
            return "trend"
        if adx <= self.regime_adx_range and bb_chg < 0:
            return "range"
        return "unknown"

    def detect_market_regime(self, adx1, adx4, adxd):
        """简易市场状态判别：根据平均ADX判断震荡或趋势"""
        adx_arr = np.array([adx1, adx4, adxd], dtype=float)
        adx_arr = adx_arr[~np.isnan(adx_arr)]
        if adx_arr.size == 0:
            return "range"
        avg_adx = adx_arr.mean()
        return "trend" if avg_adx >= 25 else "range"

    def get_ic_period_weights(self, ic_scores):
        """根据近一周 IC 加权各周期"""
        w1 = ic_scores.get("1h", 0)
        w4 = ic_scores.get("4h", 0)
        wd = ic_scores.get("d1", 0)
        w1, w4, wd = [max(v, 0) for v in (w1, w4, wd)]
        s = w1 + w4 + wd
        if s == 0:
            w1, w4, wd = 3, 2, 1
        else:
            w1 = w1 or 1e-6
            w4 = w4 or 1e-6
            wd = wd or 1e-6
        base = np.array([3, 2, 1], dtype=float)
        ic_arr = np.array([w1, w4, wd], dtype=float)
        weights = base * ic_arr
        weights = weights / weights.sum()
        return float(weights[0]), float(weights[1]), float(weights[2])

    def set_symbol_categories(self, mapping):
        """更新币种与板块的映射"""
        self.symbol_categories = {k.upper(): v for k, v in mapping.items()}


    def compute_tp_sl(
        self,
        price,
        atr,
        direction,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        *,
        rise_pred: float | None = None,
        drawdown_pred: float | None = None,
        regime: str | None = None,
    ):
        """计算止盈止损价格，可根据模型预测值微调"""
        if direction == 0:
            return None, None
        if price is None or price <= 0:
            return None, None  # 价格异常直接放弃
        if atr is None or atr == 0:
            atr = 0.005 * price

        cfg = getattr(self, "tp_sl_cfg", {})
        range_cfg = get_cfg_value(cfg, "range", {})
        trend_cfg = get_cfg_value(cfg, "trend", {})
        sl_min_pct = get_cfg_value(cfg, "sl_min_pct", 0.005)

        if regime == "range":
            tp_mult = get_cfg_value(range_cfg, "tp_mult", 1.0)
            sl_mult = get_cfg_value(range_cfg, "sl_mult", 0.8)
        elif regime == "trend":
            tp_mult = get_cfg_value(trend_cfg, "tp_mult", 1.8)
            sl_mult = get_cfg_value(trend_cfg, "sl_mult", 1.2)

        # 限制倍数范围，防止 ATR 极端波动导致止盈/止损过远或过近
        tp_mult = float(np.clip(tp_mult, 0.5, 3.0))
        sl_mult = float(np.clip(sl_mult, 0.5, 2.0))

        if rise_pred is not None and drawdown_pred is not None:
            if direction == 1:
                take_profit = price * (1 + max(rise_pred, 0))
                stop_loss = price * (1 + min(drawdown_pred, 0))
            else:
                take_profit = price * (1 - max(drawdown_pred, 0))
                stop_loss = price * (1 - min(rise_pred, 0))

            # 若预测值过小导致 tp/sl 等于入场价，退回 ATR 模式
            if abs(take_profit - price) < 1e-8 and abs(stop_loss - price) < 1e-8:
                if direction == 1:
                    take_profit = price + tp_mult * atr
                    stop_loss = price - sl_mult * atr
                else:
                    take_profit = price - tp_mult * atr
                    stop_loss = price + sl_mult * atr
        else:
            if direction == 1:
                take_profit = price + tp_mult * atr
                stop_loss = price - sl_mult * atr
            else:
                take_profit = price - tp_mult * atr
                stop_loss = price + sl_mult * atr

        min_sl_dist = max(sl_min_pct * price, 0.7 * atr)
        if direction == 1:
            if price - stop_loss < min_sl_dist:
                stop_loss = price - min_sl_dist
        else:
            if stop_loss - price < min_sl_dist:
                stop_loss = price + min_sl_dist

        return float(take_profit), float(stop_loss)

    def _base_key(self, k: str) -> str:
        for key in self.delta_params:
            if k.startswith(key):
                return key
        return k.split('_', 1)[0]

    def _calc_deltas(self, curr: dict, prev: dict, keys: list) -> dict:
        """根据配置计算关键指标变化量"""
        deltas = {}
        if prev is None:
            return {f"{k}_delta": 0.0 for k in keys}
        for k in keys:
            base = self._base_key(k)
            th, scale, _ = self.delta_params.get(base, (0, 1, 0))
            delta_raw = curr.get(k, 0) - prev.get(k, 0)
            deltas[f"{k}_delta"] = (
                delta_raw * scale if abs(delta_raw) >= th else 0.0
            )
        return deltas

    def _apply_delta_boost(self, score: float, deltas: dict) -> float:
        """根据指标变化量对得分进行微调"""
        boost = 0.0
        for k, val in deltas.items():
            if val == 0:
                continue
            base = self._base_key(k)
            _, _, inc = self.delta_params.get(base, (0, 1, 0))
            boost += np.clip(inc * np.sign(val), -0.06, 0.06)
        return score * (1 + boost)

    def _get_symbol_cache(self, symbol):
        with self._lock:
            if not symbol:
                return {
                    "history_scores": self.history_scores,
                    "oi_change_history": self.oi_change_history,
                    "_raw_history": self._raw_history,
                    "_prev_raw": self._prev_raw,
                }
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {
                    "history_scores": deque(maxlen=self.history_scores.maxlen),
                    "oi_change_history": deque(maxlen=self.oi_change_history.maxlen),
                    "_raw_history": {
                        "15m": deque(maxlen=4),
                        "1h": deque(maxlen=4),
                        "4h": deque(maxlen=2),
                        "d1": deque(maxlen=2),
                    },
                    "_prev_raw": {p: None for p in ("15m", "1h", "4h", "d1")},
                }
            return self.symbol_data[symbol]

    def _normalize_features(self, feats, period: str) -> dict:
        """将 DataFrame/Series 输入转为字典, 并缓存列索引"""
        if isinstance(feats, dict):
            return feats
        if isinstance(feats, pd.Series):
            cols = tuple(feats.index)
            cache = self._std_index_cache.get(period)
            if cache is None or cache[0] != cols:
                idx_map = {c: i for i, c in enumerate(cols)}
                self._std_index_cache[period] = (cols, idx_map)
            else:
                idx_map = cache[1]
            return {c: feats.iat[i] for c, i in idx_map.items()}
        if isinstance(feats, pd.DataFrame) and not feats.empty:
            row = feats.iloc[-1]
            cols = tuple(row.index)
            cache = self._std_index_cache.get(period)
            if cache is None or cache[0] != cols:
                idx_map = {c: i for i, c in enumerate(cols)}
                self._std_index_cache[period] = (cols, idx_map)
            else:
                idx_map = cache[1]
            return {c: row.iat[i] for c, i in idx_map.items()}
        return {}

    def ma_cross_logic(self, features: dict, sma_20_1h_prev=None) -> float:
        """根据1h MA5 与 MA20 判断并返回分数乘数"""

        sma5 = features.get('sma_5_1h')
        sma20 = features.get('sma_20_1h')
        ma_ratio = features.get('ma_ratio_5_20', 1.0)
        if sma5 is None or sma20 is None:
            return 1.0

        slope = 0.0
        if sma_20_1h_prev not in (None, 0):
            slope = (sma20 - sma_20_1h_prev) / sma_20_1h_prev

        if (ma_ratio > 1.02 and slope > 0) or (ma_ratio < 0.98 and slope < 0):
            return 1.15
        if (ma_ratio > 1.02 and slope < 0) or (ma_ratio < 0.98 and slope > 0):
            return 0.85
        return 1.0

    def detect_reversal(
        self,
        price_series,
        atr,
        volume,
        win: int = 3,
        atr_mult: float = 1.05,
        vol_mult: float = 1.10,
    ) -> int:
        """V 型急反转检测"""

        if len(price_series) < win + 1 or atr is None:
            return 0
        pct = np.diff(price_series) / price_series[:-1]
        slope_now, slope_prev = pct[-1], pct[-win:].mean()
        amp = max(price_series[-win - 1 :]) - min(price_series[-win - 1 :])
        price_base = price_series[-2] or price_series[-1]
        amp_pct = amp / price_base if price_base else 0
        cond_amp = amp_pct > atr_mult * atr
        cond_vol = (volume is None) or (volume >= vol_mult)
        if np.sign(slope_now) != np.sign(slope_prev) and cond_amp and cond_vol:
            return int(np.sign(slope_now))
        return 0


    def compute_exit_multiplier(self, vote: float, prev_vote: float, last_signal: int) -> float:
        """根据票数变化决定半退出或全平仓位系数"""

        with self._lock:
            exit_lag = self._exit_lag

        exit_mult = 1.0
        vote_sign = np.sign(vote)
        prev_sign = np.sign(prev_vote)
        if last_signal == 1:
            if vote_sign == 1 and prev_vote > vote:
                exit_mult = 0.5
                exit_lag = 0
            elif vote_sign <= 0 and prev_sign > 0:
                exit_lag += 1
                exit_mult = 0.0 if exit_lag >= self.exit_lag_bars else 0.5
            else:
                exit_lag = 0
        elif last_signal == -1:
            if vote_sign == -1 and prev_vote < vote:
                exit_mult = 0.5
                exit_lag = 0
            elif vote_sign >= 0 and prev_sign < 0:
                exit_lag += 1
                exit_mult = 0.0 if exit_lag >= self.exit_lag_bars else 0.5
            else:
                exit_lag = 0
        else:
            exit_lag = 0

        with self._lock:
            self._exit_lag = exit_lag

        return exit_mult

    def compute_position_size(
        self,
        *,
        grad_dir: float,
        base_coeff: float,
        confidence_factor: float,
        vol_ratio: float | None,
        fused_score: float,
        base_th: float,
        regime: str,
        oi_overheat: bool,
        vol_p: float | None,
        risk_score: float,
        crowding_factor: float,
        cfg_th_sig: dict,
        scores: dict,
        direction: int,
        exit_mult: float,
        consensus_all: bool = False,
    ) -> tuple[float, int, float, str | None]:
        """Calculate final position size and tier given direction and risk factors.

        Also return a ``zero_reason`` when ``pos_size`` is reduced to zero so the
        caller can trace why no position will be taken.
        """

        tier = base_coeff * abs(grad_dir)
        base_size = tier

        zero_reason: str | None = None
        low_vol_flag = False

        risk_factor = 1.0 / (1.0 + risk_score)
        pos_size = base_size * sigmoid(confidence_factor) * risk_factor
        pos_size *= exit_mult
        pos_size = min(pos_size, self.max_position)
        pos_size *= crowding_factor
        if direction == 0:
            pos_size = 0.0
            zero_reason = "no_direction"

        if (
            regime == "range"
            and vol_ratio is not None
            and vol_ratio < self.low_vol_ratio
            and abs(fused_score) < base_th + 0.02
            and not consensus_all
        ):
            pos_size *= 0.5
            low_vol_flag = True


        if vol_p is not None:
            pos_size *= max(0.4, 1 - min(0.6, vol_p))

        # ↓ 允许极小仓位，交由风险控制模块再裁剪
        min_pos = cfg_th_sig.get("min_pos", self.signal_params.min_pos)
        if pos_size < min_pos:
            direction, pos_size = 0, 0.0
            if low_vol_flag:
                zero_reason = "vol_ratio"
            else:
                zero_reason = "min_pos"

        # 4h 周期 veto 逻辑已停用
        # if direction == 1 and scores.get("4h", 0) < -self.veto_level:
        #     direction, pos_size = 0, 0.0
        # elif direction == -1 and scores.get("4h", 0) > self.veto_level:
        #     direction, pos_size = 0, 0.0

        return pos_size, direction, tier, zero_reason

    # >>>>> 修改：改写 get_ai_score，让它自动从 self.models[...]["features"] 中取“训练时列名”
    def get_ai_score(self, features, model_up, model_down, calibrator_up=None, calibrator_down=None):
        """根据上下模型概率计算AI得分"""
        return self.ai_predictor.get_ai_score(features, model_up, model_down, calibrator_up, calibrator_down)

    def get_ai_score_cls(self, features, model_dict):
        """从单个分类模型计算AI得分"""
        return self.ai_predictor.get_ai_score_cls(features, model_dict)

    def get_vol_prediction(self, features, model_dict):
        """根据回归模型预测未来波动率"""
        return self.ai_predictor.get_vol_prediction(features, model_dict)

    def get_reg_prediction(self, features, model_dict):
        """通用回归模型预测"""
        return self.ai_predictor.get_reg_prediction(features, model_dict)

    # robust_signal_generator.py

    def get_factor_scores(self, features: dict, period: str) -> dict:
        """
        输入：
          - features: 单周期特征字典（如 {'ema_diff_1h': 0.12, 'boll_perc_1h': 0.45, ...}）
          - period:   "1h" / "4h" / "d1"
        输出：一个 dict，包含6个子因子得分。
        """

        key = self._make_cache_key(features, period)
        cached = self._cache_get(self._factor_cache, key)
        if cached is not None:
            return cached

        # 去除重复字段，避免两次写入同名特征
        dedup_row = {k: v for k, v in features.items()}

        def safe(key: str, default=0):
            """如果值缺失或为 NaN，返回 default。"""
            v = dedup_row.get(key, default)
            if v is None:
                return default
            if isinstance(v, (float, int)) and pd.isna(v):
                return default
            return v

        trend_raw = (
            np.tanh(safe(f'ema_diff_{period}', 0) * 5)
            + 2 * (safe(f'boll_perc_{period}', 0.5) - 0.5)
            + safe(f'supertrend_dir_{period}', 0)
            + np.tanh(safe(f'adx_delta_{period}', 0) / 10)
            + np.tanh((safe(f'bull_streak_{period}', 0) - safe(f'bear_streak_{period}', 0)) / 3)
            + 0.5 * safe(f'long_lower_shadow_{period}', 0)
            - 0.5 * safe(f'long_upper_shadow_{period}', 0)
            + 0.5 * safe(f'vol_breakout_{period}', 0)
            + np.tanh(safe(f'ichimoku_cloud_thickness_{period}', 0))
            + np.tanh((safe(f'close_{period}', safe("close", 0)) / safe(f'vwap_{period}', 1) - 1) * 5)
            + np.tanh((safe(f'kc_perc_{period}', 0.5) - 0.5) * 3)
            + np.tanh((safe(f'donchian_perc_{period}', 0.5) - 0.5) * 3)
            + 0.5 * np.tanh(
                (safe(f'ichimoku_conversion_{period}', 0) - safe(f'ichimoku_base_{period}', 0))
                / (abs(safe(f'close_{period}', safe("close", 1))) + 1e-6)
                * 10
            )
            + 0.5 * np.tanh(
                (
                    safe(f'sma_10_{period}', safe(f'sma_20_{period}', 0))
                    / (safe(f'sma_20_{period}', 1) or 1)
                    - 1
                )
                * 5
            )
            + 0.3 * np.tanh(safe('close_spread_1h_4h', 0) * 5)
            + 0.3 * np.tanh(safe('close_spread_1h_d1', 0) * 5)
            + np.tanh(safe(f"close_vs_pivot_{period}", 0) * 8)
        )

        momentum_raw = (
            (safe(f'rsi_{period}', 50) - 50) / 50
            + safe(f'willr_{period}', -50) / 50
            + np.tanh(safe(f'macd_hist_{period}', 0) * 5)
            + np.tanh(safe(f'rsi_slope_{period}', 0) * 10)
            + (safe(f'mfi_{period}', 50) - 50) / 50
            + 0.5 * np.tanh(safe('rsi_diff_1h_4h', 0) / 10)
            + 0.5 * np.tanh(safe('rsi_diff_1h_d1', 0) / 10)
            + 0.5 * np.tanh(safe('rsi_diff_4h_d1', 0) / 10)
            + 0.5 * np.tanh((safe(f'stoch_k_{period}', 50) - 50) / 50)
            + 0.5 * np.tanh((safe(f'stoch_d_{period}', 50) - 50) / 50)
            + 0.5 * np.tanh(safe(f'macd_signal_{period}', 0) * 5)
            + 0.3 * np.tanh(safe(f'pct_chg1_{period}', 0) * 20)
            + 0.2 * np.tanh(safe(f'pct_chg3_{period}', 0) * 10)
            + 0.2 * np.tanh(safe(f'pct_chg6_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'cci_{period}', 0) / 100)
            + 0.3 * np.tanh(safe(f'cci_delta_{period}', 0) / 20)
            + 0.3 * np.tanh(safe('macd_hist_4h_mul_bb_width_1h', 0) * 5)
        )

        volatility_raw = (
            np.tanh(safe(f'atr_pct_{period}', 0) * 8)
            + np.tanh(safe(f'bb_width_{period}', 0) * 2)
            + np.tanh(safe(f'donchian_delta_{period}', 0) * 5)
            + np.tanh(safe(f'hv_7d_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'hv_14d_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'hv_30d_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'kc_width_pct_chg_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'skewness_{period}', 0) * 5)
            + 0.5 * np.tanh((safe(f'kurtosis_{period}', 3) - 3))
            + 0.5 * np.tanh(safe(f'atr_chg_{period}', 0) * 50)
            + 0.5 * np.tanh(safe(f'bb_width_chg_{period}', 0) * 20)
            + 0.3 * np.tanh(safe('mom_5m_roll1h_std', 0) * 5)
            + 0.3 * np.tanh(safe('mom_15m_roll1h_std', 0) * 5)
        )

        volume_raw = (
            np.tanh(safe(f'vol_ma_ratio_{period}', 0))
            + np.tanh(safe(f'obv_delta_{period}', 0) / 1e5)
            + np.tanh(safe(f'vol_roc_{period}', 0) / 5)
            + np.tanh(safe(f'rsi_mul_vol_ma_ratio_{period}', 0) / 100)
            + np.tanh((safe(f'buy_sell_ratio_{period}', 1) - 1) * 2)
            + np.tanh(safe(f'vol_profile_density_{period}', 0) / 10)
            + np.tanh((safe(f'money_flow_ratio_{period}', 1) - 1) * 2)
            - np.tanh(safe(f'bid_ask_spread_pct_{period}', 0) * 10)
            + 0.5 * np.tanh((safe(f'vol_ma_ratio_long_{period}', 1) - 1) * 2)
            + 0.5 * np.tanh(safe(f'cg_total_volume_roc_{period}', 0) * 5)
            + 0.5 * np.tanh(safe('bid_ask_imbalance', 0) * 10)
            + 0.3 * np.tanh(safe('vol_ratio_4h_d1', 0))
            + 0.3 * np.tanh(safe('rsi_1h_mul_vol_ma_ratio_4h', 0) / 100)
            + np.tanh(safe(f"close_vs_vpoc_{period}", 0) * 8)
        )

        sentiment_raw = (
            (safe('fg_index_d1', 50) - 50) / 50
            + np.tanh(safe(f'btc_correlation_1h_{period}', 0))
            + np.tanh(safe(f'eth_correlation_1h_{period}', 0))
            + np.tanh(safe(f'price_diff_cg_{period}', 0) * 5)
            + np.tanh(safe(f'cg_market_cap_roc_{period}', 0) * 5)
            + np.tanh((safe(f'volume_cg_ratio_{period}', 1) - 1) * 2)
            + 0.5 * np.tanh((safe(f'price_ratio_cg_{period}', 1) - 1) * 10)
        )

        f_rate = safe(f'funding_rate_{period}', 0)
        f_anom = safe(f'funding_rate_anom_{period}', 0)
        thr = 0.0005  # 約 0.05% 年化
        if abs(f_rate) > thr:
            funding_raw = -np.tanh(f_rate * 4000)  # 4000 ≈ 1/0.00025，讓 ±0.002 ≈ tanh(8)
        else:
            funding_raw = np.tanh(f_rate * 4000)
        if abs(f_rate) < 0.001:
            funding_raw = 0.0
        funding_raw += np.tanh(f_anom * 50)
        scores = {
            'trend': np.tanh(trend_raw),
            'momentum': np.tanh(momentum_raw),
            'volatility': np.tanh(volatility_raw),
            'volume': np.tanh(volume_raw),
            'sentiment': np.tanh(sentiment_raw),
            'funding': np.tanh(funding_raw),
        }

        pos = safe(f'channel_pos_{period}', 0.5)
        for k, v in scores.items():
            if pos > 1 and v > 0:
                scores[k] = v * 1.2
            elif pos < 0 and v < 0:
                scores[k] = v * 1.2
            elif pos > 0.9 and v > 0:
                scores[k] = v * 0.8
            elif pos < 0.1 and v < 0:
                scores[k] = v * 0.8

        self._cache_set(self._factor_cache, key, scores)
        return scores

    def update_ic_scores(self, df, *, window=None, group_by=None, time_col="open_time"):
        """根据历史数据计算并更新各因子的 IC 分数

        Parameters
        ----------
        df : pandas.DataFrame
            历史特征数据。
        window : int, optional
            只取最近 ``window`` 条记录参与计算；默认为 ``None`` 表示使用全部数据。
        group_by : str, optional
            若指定，对 ``df`` 按该列分组后分别计算 IC，再取平均值。
        time_col : str, default ``"open_time"``
            排序所依据的时间列名。
        """

        from quant_trade.param_search import compute_ic_scores

        def _compute(sub_df: pd.DataFrame) -> dict:
            sub_df = sub_df.sort_values(time_col)
            if window:
                sub_df = sub_df.tail(window)
            return compute_ic_scores(sub_df, self)

        if group_by:
            grouped = df.groupby(group_by)
            ic_list = []
            for _, g in grouped:
                ic_list.append(_compute(g))
            if ic_list:
                ic = {k: float(np.nanmean([d[k] for d in ic_list])) for k in self.base_weights}
            else:
                ic = {k: 0.0 for k in self.base_weights}
        else:
            ic = _compute(df)

        self.ic_scores.update(ic)

        if not hasattr(self, "ic_history"):
            self.ic_history = {k: deque(maxlen=500) for k in self.base_weights}

        with self._lock:
            for k, v in ic.items():
                self.ic_history.setdefault(k, deque(maxlen=500)).append(v)

        return self.ic_scores

    def dynamic_weight_update(self, halflife=20):
        """根据因子IC的指数加权均值更新权重"""
        with self._lock:
            if not hasattr(self, "ic_history"):
                self.ic_history = {k: deque(maxlen=3000) for k in self.base_weights}

            ic_avg = []
            decay = np.log(0.5) / float(halflife)
            for k in self.ic_scores.keys():
                hist = self.ic_history.get(k)
                if hist:
                    arr = np.array(hist, dtype=float)
                    weights = np.exp(decay * np.arange(len(arr))[::-1])
                    weights /= weights.sum()
                    ic_avg.append(float(np.nansum(arr * weights)))
                else:
                    ic_avg.append(self.ic_scores[k])

            raw = {}
            for k, ic_val in zip(self.ic_scores.keys(), ic_avg):
                base_w = self.base_weights.get(k, 0)
                if ic_val < 0:
                    w = base_w * max(0.0, 1 - abs(ic_val))
                else:
                    w = base_w * (1 + ic_val)
                raw[k] = max(base_w * self.min_weight_ratio, w)

        total = sum(raw.values()) or 1.0
        self.current_weights = {k: v / total for k, v in raw.items()}
        logger.info("current_weights: %s", self.current_weights)
        return self.current_weights

    def _weight_update_loop(self, interval):
        while True:
            try:
                self.dynamic_weight_update()
            except Exception as e:
                logger.warning("weight update failed: %s", e)
            time.sleep(interval)

    def start_weight_update_thread(self, interval=300):
        t = threading.Thread(target=self._weight_update_loop, args=(interval,), daemon=True)
        t.start()
        self._weight_thread = t

    def compute_dynamic_threshold(
        self,
        data: DynamicThresholdInput,
        *,
        base: float | None = None,
        low_base: float | None = None,
        history_scores=None,
    ):
        """Calculate dynamic threshold using provided metrics."""

        params = self.signal_params
        base = params.base_th if base is None else base
        low_base = params.low_base if low_base is None else low_base

        hist_base = _calc_history_base(
            history_scores,
            base,
            params.quantile,
            self.th_window,
            self.th_decay,
            0.12,
        ) if history_scores is not None else base

        th = hist_base
        atr_eff = abs(data.atr)
        if data.atr_4h is not None:
            atr_eff += 0.5 * abs(data.atr_4h)
        if data.atr_d1 is not None:
            atr_eff += 0.25 * abs(data.atr_d1)
        th += min(0.10, atr_eff * params.atr_mult)

        fund_eff = abs(data.funding)
        if data.pred_vol is not None:
            fund_eff += 0.5 * abs(data.pred_vol)
        if data.pred_vol_4h is not None:
            fund_eff += 0.25 * abs(data.pred_vol_4h)
        if data.pred_vol_d1 is not None:
            fund_eff += 0.15 * abs(data.pred_vol_d1)
        if data.vix_proxy is not None:
            fund_eff += 0.25 * abs(data.vix_proxy)
        th += min(0.08, fund_eff * params.funding_mult)

        adx_eff = abs(data.adx)
        if data.adx_4h is not None:
            adx_eff += 0.5 * abs(data.adx_4h)
        if data.adx_d1 is not None:
            adx_eff += 0.25 * abs(data.adx_d1)
        th += min(0.04, adx_eff / params.adx_div)

        if atr_eff == 0 and adx_eff == 0 and fund_eff == 0:
            th = min(th, hist_base)

        if data.reversal:
            th *= params.rev_th_mult

        rev_boost = params.rev_boost
        if data.regime == "trend":
            th *= 1.05
            rev_boost *= 0.8
        elif data.regime == "range":
            th *= 0.95
            rev_boost *= 1.2

        return max(th, low_base), rev_boost

    # Backward compatible wrapper
    def dynamic_threshold(
        self,
        atr,
        adx,
        funding=0,
        atr_4h=None,
        adx_4h=None,
        atr_d1=None,
        adx_d1=None,
        pred_vol=None,
        pred_vol_4h=None,
        pred_vol_d1=None,
        vix_proxy=None,
        base=0.08,
        regime=None,
        low_base=None,
        reversal=False,
        history_scores=None,
    ):
        data = DynamicThresholdInput(
            atr=atr,
            adx=adx,
            funding=funding,
            atr_4h=atr_4h,
            adx_4h=adx_4h,
            atr_d1=atr_d1,
            adx_d1=adx_d1,
            pred_vol=pred_vol,
            pred_vol_4h=pred_vol_4h,
            pred_vol_d1=pred_vol_d1,
            vix_proxy=vix_proxy,
            regime=regime,
            reversal=reversal,
        )
        return self.compute_dynamic_threshold(
            data,
            base=base,
            low_base=low_base,
            history_scores=history_scores,
        )

    def combine_score(self, ai_score, factor_scores, weights=None):
        """按固定顺序加权合并各因子得分"""
        if weights is None:
            weights = self.base_weights

        fused_score = (
            ai_score * weights['ai']
            + factor_scores['trend'] * weights['trend']
            + factor_scores['momentum'] * weights['momentum']
            + factor_scores['volatility'] * weights['volatility']
            + factor_scores['volume'] * weights['volume']
            + factor_scores['sentiment'] * weights['sentiment']
            + factor_scores['funding'] * weights['funding']
        )

        return float(fused_score)

    def combine_score_vectorized(self, ai_scores, factor_scores, weights=None):
        """向量化计算多个样本的合并得分"""
        if weights is None:
            weights = self.base_weights

        weight_arr = np.array(
            [
                weights['ai'],
                weights['trend'],
                weights['momentum'],
                weights['volatility'],
                weights['volume'],
                weights['sentiment'],
                weights['funding'],
            ],
            dtype=float,
        )

        fs_matrix = np.vstack(
            [
                ai_scores,
                factor_scores['trend'],
                factor_scores['momentum'],
                factor_scores['volatility'],
                factor_scores['volume'],
                factor_scores['sentiment'],
                factor_scores['funding'],
            ]
        )

        return (fs_matrix.T * weight_arr).sum(axis=1).astype(float)

    def consensus_check(self, s1, s2, s3, min_agree=2):
        # 多周期方向共振（如调研建议），可加全分歧减弱等逻辑
        signs = np.sign([s1, s2, s3])
        non_zero = [g for g in signs if g != 0]
        if len(non_zero) < min_agree:
            return 0  # 无方向共振
        cnt = Counter(non_zero)
        if cnt.most_common(1)[0][1] >= min_agree:
            return int(cnt.most_common(1)[0][0])  # 返回方向
        return int(np.sign(np.sum(signs)))

    def crowding_protection(self, scores, current_score, base_th=0.2):
        """根据同向排名抑制过度拥挤的信号，返回衰减系数"""
        if not scores or len(scores) < 30:
            return 1.0

        arr = np.array(scores, dtype=float)
        mask = np.abs(arr) >= base_th * 0.8
        arr = arr[mask]
        signs = [s for s in np.sign(arr) if s != 0]
        total = len(signs)
        if total == 0:
            return 1.0
        pos_counts = Counter(signs)
        dominant_dir, cnt = pos_counts.most_common(1)[0]
        if np.sign(current_score) != dominant_dir:
            return 1.0

        ratio = cnt / total
        rank_pct = pd.Series(np.abs(list(arr) + [current_score])).rank(pct=True).iloc[-1]
        ratio_intensity = max(0.0, (ratio - self.max_same_direction_rate) / (1 - self.max_same_direction_rate))
        rank_intensity = max(0.0, rank_pct - 0.8) / 0.2
        intensity = min(1.0, max(ratio_intensity, rank_intensity))

        factor = 1.0 - 0.2 * intensity
        dd = getattr(self, "_equity_drawdown", 0.0)
        factor *= max(0.6, 1 - dd)
        return factor

    def apply_oi_overheat_protection(self, fused_score, oi_chg, th_oi):
        """Adjust score based on open interest change."""
        if th_oi is None or abs(oi_chg) < th_oi:
            # Mild change: slightly reward or penalise according to oi_chg
            return fused_score * (1 + 0.03 * oi_chg), False

        logging.info("OI overheat detected: %.4f", oi_chg)
        # Only scale down when overheating
        return fused_score * self.oi_scale, True

    # ===== 新增辅助函数 =====
    def calc_factor_scores(self, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
        """计算未调整的各周期得分"""
        w1 = weights.copy()
        w4 = weights.copy()
        for k in ('trend', 'momentum', 'volume'):
            w1[k] = w1.get(k, 0) * 0.7
            w4[k] = w4.get(k, 0) * 0.7
        scores = {
            '1h': self.combine_score(ai_scores['1h'], factor_scores['1h'], w1),
            '4h': self.combine_score(ai_scores['4h'], factor_scores['4h'], w4),
            'd1': self.combine_score(ai_scores['d1'], factor_scores['d1'], weights),
        }
        logger.debug("factor scores: %s", scores)
        return scores

    def calc_factor_scores_vectorized(
        self,
        ai_scores: dict,
        factor_scores: dict,
        weights: dict,
    ) -> dict:
        """向量化版本的各周期得分计算"""

        w1 = weights.copy()
        w4 = weights.copy()
        for k in ('trend', 'momentum', 'volume'):
            w1[k] = w1.get(k, 0) * 0.7
            w4[k] = w4.get(k, 0) * 0.7

        return {
            '1h': self.combine_score_vectorized(ai_scores['1h'], factor_scores['1h'], w1),
            '4h': self.combine_score_vectorized(ai_scores['4h'], factor_scores['4h'], w4),
            'd1': self.combine_score_vectorized(ai_scores['d1'], factor_scores['d1'], weights),
        }

    def apply_local_adjustments(
        self,
        scores: dict,
        raw_feats: dict,
        factor_scores: dict,
        deltas: dict,
        rise_pred_1h: float | None = None,
        drawdown_pred_1h: float | None = None,
    ) -> tuple[dict, dict]:
        """应用本地逻辑修正分数并返回细节"""

        adjusted = scores.copy()
        details = {}

        for p in adjusted:
            adjusted[p] = self._apply_delta_boost(adjusted[p], deltas.get(p, {}))

        prev_ma20 = raw_feats['1h'].get('sma_20_1h_prev')
        ma_coeff = self.ma_cross_logic(raw_feats['1h'], prev_ma20)
        adjusted['1h'] *= ma_coeff
        details['ma_cross'] = int(np.sign(ma_coeff - 1.0))

        if rise_pred_1h is not None and drawdown_pred_1h is not None:
            adj = np.tanh((rise_pred_1h - abs(drawdown_pred_1h)) * 5) * 0.5
            adjusted['1h'] *= 1 + adj
            details['rise_drawdown_adj'] = adj

        strong_confirm_4h = (
            factor_scores['4h']['trend'] > 0
            and factor_scores['4h']['momentum'] > 0
            and factor_scores['4h']['volatility'] > 0
            and adjusted['4h'] > 0
        ) or (
            factor_scores['4h']['trend'] < 0
            and factor_scores['4h']['momentum'] < 0
            and factor_scores['4h']['volatility'] < 0
            and adjusted['4h'] < 0
        )
        details['strong_confirm_4h'] = strong_confirm_4h

        macd_diff = raw_feats['1h'].get('macd_hist_diff_1h_4h')
        rsi_diff = raw_feats['1h'].get('rsi_diff_1h_4h')
        if (
            macd_diff is not None
            and rsi_diff is not None
            and macd_diff < 0
            and rsi_diff < -8
        ):
            if strong_confirm_4h:
                logger.debug(
                    "momentum misalign macd_diff=%.3f rsi_diff=%.3f -> strong_confirm=False",
                    macd_diff,
                    rsi_diff,
                )
            strong_confirm_4h = False
            details['strong_confirm_4h'] = False

        if (
            macd_diff is not None
            and rsi_diff is not None
            and abs(macd_diff) < 5
            and abs(rsi_diff) < 15
        ):
            strong_confirm_4h = True
            details['strong_confirm_4h'] = True

        for p in ['1h', '4h', 'd1']:
            sent = factor_scores[p]['sentiment']
            before = adjusted[p]
            adjusted[p] = adjust_score(
                adjusted[p],
                sent,
                self.sentiment_alpha,
                cap_scale=self.cap_positive_scale,
            )
            if before != adjusted[p]:
                logger.debug(
                    "sentiment %.2f adjust %s: %.3f -> %.3f",
                    sent,
                    p,
                    before,
                    adjusted[p],
                )

        params = self.volume_guard_params
        r1 = raw_feats['1h'].get('vol_ma_ratio_1h')
        roc1 = raw_feats['1h'].get('vol_roc_1h')
        before = adjusted['1h']
        adjusted['1h'] = volume_guard(adjusted['1h'], r1, roc1, **params)
        if before != adjusted['1h']:
            logger.debug(
                "volume guard 1h ratio=%.3f roc=%.3f -> %.3f",
                r1,
                roc1,
                adjusted['1h'],
            )
        if raw_feats.get('4h') is not None:
            r4 = raw_feats['4h'].get('vol_ma_ratio_4h')
            roc4 = raw_feats['4h'].get('vol_roc_4h')
            before4 = adjusted['4h']
            adjusted['4h'] = volume_guard(adjusted['4h'], r4, roc4, **params)
            if before4 != adjusted['4h']:
                logger.debug(
                    "volume guard 4h ratio=%.3f roc=%.3f -> %.3f",
                    r4,
                    roc4,
                    adjusted['4h'],
                )
        r_d1 = raw_feats['d1'].get('vol_ma_ratio_d1')
        roc_d1 = raw_feats['d1'].get('vol_roc_d1')
        before_d1 = adjusted['d1']
        adjusted['d1'] = volume_guard(adjusted['d1'], r_d1, roc_d1, **params)
        if before_d1 != adjusted['d1']:
            logger.debug(
                "volume guard d1 ratio=%.3f roc=%.3f -> %.3f",
                r_d1,
                roc_d1,
                adjusted['d1'],
            )

        for p in ['1h', '4h', 'd1']:
            bs = raw_feats[p].get(f'break_support_{p}')
            br = raw_feats[p].get(f'break_resistance_{p}')
            before_sr = adjusted[p]
            if br:
                adjusted[p] *= 1.1 if adjusted[p] > 0 else 0.8
            if bs:
                adjusted[p] *= 1.1 if adjusted[p] < 0 else 0.8
            if before_sr != adjusted[p]:
                logger.debug(
                    "break SR %s bs=%s br=%s %.3f->%.3f",
                    p,
                    bs,
                    br,
                    before_sr,
                    adjusted[p],
                )
                details[f'break_sr_{p}'] = adjusted[p] - before_sr

        for p in ['1h', '4h', 'd1']:
            perc = raw_feats[p].get(f'boll_perc_{p}')
            vol_ratio = raw_feats[p].get(f'vol_ma_ratio_{p}')
            before_bb = adjusted[p]
            if (
                perc is not None
                and vol_ratio is not None
                and vol_ratio > 1.5
                and (perc >= 0.98 or perc <= 0.02)
            ):
                if perc >= 0.98:
                    adjusted[p] *= 1.1 if adjusted[p] > 0 else 0.9
                else:
                    adjusted[p] *= 1.1 if adjusted[p] < 0 else 0.9
            if before_bb != adjusted[p]:
                logger.debug(
                    "boll breakout %s perc=%.3f vol_ratio=%.3f %.3f->%.3f",
                    p,
                    perc,
                    vol_ratio,
                    before_bb,
                    adjusted[p],
                )
                details[f'boll_breakout_{p}'] = adjusted[p] - before_bb

        return adjusted, details

    def fuse_multi_cycle(
        self,
        scores: dict,
        weights: tuple[float, float, float],
        strong_confirm_4h: bool,
    ) -> tuple[float, bool, bool, bool]:
        """按照多周期共振逻辑融合得分"""
        s1, s4, sd = scores['1h'], scores['4h'], scores['d1']
        w1, w4, wd = weights

        consensus_dir = self.consensus_check(s1, s4, sd)
        consensus_all = consensus_dir != 0 and np.sign(s1) == np.sign(s4) == np.sign(sd)
        consensus_14 = consensus_dir != 0 and np.sign(s1) == np.sign(s4) and not consensus_all
        consensus_4d1 = consensus_dir != 0 and np.sign(s4) == np.sign(sd) and np.sign(s1) != np.sign(s4)

        if consensus_all:
            fused = w1 * s1 + w4 * s4 + wd * sd
            conf = 1.0
            if strong_confirm_4h:
                fused *= 1.15
            fused *= self.cycle_weight.get("strong", 1.0)
        elif consensus_14:
            total = w1 + w4
            fused = (w1 / total) * s1 + (w4 / total) * s4
            conf = 0.8
            if strong_confirm_4h:
                fused *= 1.10
            fused *= self.cycle_weight.get("weak", 1.0)
        elif consensus_4d1:
            total = w4 + wd
            fused = (w4 / total) * s4 + (wd / total) * sd
            conf = 0.7
            fused *= self.cycle_weight.get("weak", 1.0)
        else:
            fused = s1
            conf = 0.6

        fused_score = fused * conf
        if (
            np.sign(s1) != 0
            and (
                (np.sign(s4) != 0 and np.sign(s1) != np.sign(s4))
                or (np.sign(sd) != 0 and np.sign(s1) != np.sign(sd))
            )
        ):
            fused_score *= self.cycle_weight.get("opposite", 1.0)
        logger.debug(
            "fuse scores s1=%.3f s4=%.3f sd=%.3f -> %.3f",
            s1,
            s4,
            sd,
            fused_score,
        )
        return fused_score, consensus_all, consensus_14, consensus_4d1

    # ===== 新增私有方法 =====

    def _to_hashable(self, obj):
        """将输入对象转换为可哈希的形式，用于缓存键"""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._to_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(self._to_hashable(v) for v in obj)
        if isinstance(obj, np.generic):
            return float(obj)
        return obj

    def _make_cache_key(self, *objs):
        return tuple(self._to_hashable(o) if o is not None else None for o in objs)

    def _cache_get(self, cache: OrderedDict, key):
        with self._lock:
            return cache.get(key)

    def _cache_set(self, cache: OrderedDict, key, value):
        with self._lock:
            cache[key] = value
            if len(cache) > self.cache_maxsize:
                cache.popitem(last=False)

    def _normalize_inputs(
        self,
        features_1h,
        features_4h,
        features_d1,
        features_15m=None,
        raw_features_1h=None,
        raw_features_4h=None,
        raw_features_d1=None,
        raw_features_15m=None,
    ):
        """规范化输入特征与原始特征"""
        f1h = self._normalize_features(features_1h, "1h")
        f4h = self._normalize_features(features_4h, "4h")
        fd1 = self._normalize_features(features_d1, "d1")
        f15m = self._normalize_features(features_15m or {}, "15m")

        r1h = self._normalize_features(raw_features_1h or {}, "1h")
        r4h = self._normalize_features(raw_features_4h or {}, "4h")
        rd1 = self._normalize_features(raw_features_d1 or {}, "d1")
        r15m = self._normalize_features(raw_features_15m or {}, "15m")

        return f1h, f4h, fd1, f15m, r1h, r4h, rd1, r15m

    def compute_ai_scores(self, std_1h, std_4h, std_d1, raw_fd1):
        """封装 AI 模型推理与校准"""
        key = self._make_cache_key(std_1h, std_4h, std_d1, raw_fd1)
        cached = self._cache_get(self._ai_score_cache, key)
        if cached is not None:
            return cached

        ai_scores: dict[str, float] = {}
        vol_preds: dict[str, float | None] = {}
        rise_preds: dict[str, float | None] = {}
        drawdown_preds: dict[str, float | None] = {}

        for p, feats in [("1h", std_1h), ("4h", std_4h), ("d1", std_d1)]:
            models_p = self.models.get(p, {})
            if "cls" in models_p and "up" not in models_p:
                ai_scores[p] = self.get_ai_score_cls(feats, models_p["cls"])
            else:
                cal_up = self.calibrators.get(p, {}).get("up")
                cal_down = self.calibrators.get(p, {}).get("down")
                if cal_up is None and cal_down is None:
                    ai_scores[p] = self.get_ai_score(
                        feats,
                        models_p["up"],
                        models_p["down"],
                    )
                else:
                    ai_scores[p] = self.get_ai_score(
                        feats,
                        models_p["up"],
                        models_p["down"],
                        cal_up,
                        cal_down,
                    )
            if "vol" in models_p:
                vol_preds[p] = self.get_vol_prediction(feats, models_p["vol"])
            if "rise" in models_p:
                rise_preds[p] = self.get_reg_prediction(feats, models_p["rise"])
            if "drawdown" in models_p:
                drawdown_preds[p] = self.get_reg_prediction(
                    feats, models_p["drawdown"]
                )

        oversold_reversal = False
        rsi = raw_fd1.get("rsi_d1", 50)
        cci = raw_fd1.get("cci_d1", 0)
        if rsi < 25 or cci < -100 or rsi > 75 or cci > 100:
            ai_scores["d1"] *= 0.7
            oversold_reversal = True

        if ai_scores.get("d1", 0) < 0 and abs(ai_scores["d1"]) < self.th_down_d1:
            ai_scores["d1"] = 0.0

        result = ai_scores, vol_preds, rise_preds, drawdown_preds, oversold_reversal
        self._cache_set(self._ai_score_cache, key, result)
        return result

    def compute_factor_scores(
        self,
        ai_scores: dict,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        std_15m: dict,
        raw_dict: dict,
        deltas: dict,
        rise_preds: dict,
        drawdown_preds: dict,
        vol_preds: dict,
        global_metrics: dict | None,
        open_interest: dict | None,
        ob_imb,
        symbol: str | None,
    ):
        """计算多因子得分并输出相关中间结果"""
        key = self._make_cache_key(
            ai_scores,
            std_1h,
            std_4h,
            std_d1,
            std_15m,
            raw_dict,
            deltas,
            rise_preds,
            drawdown_preds,
            vol_preds,
            global_metrics,
            open_interest,
            ob_imb,
            symbol,
        )
        cached = self._cache_get(self._factor_score_cache, key)
        if cached is not None:
            return cached
        fs = {
            "1h": self.get_factor_scores(std_1h, "1h"),
            "4h": self.get_factor_scores(std_4h, "4h"),
            "d1": self.get_factor_scores(std_d1, "d1"),
        }

        with self._lock:
            weights = self.current_weights.copy()

        scores = self.calc_factor_scores(ai_scores, fs, weights)

        scores, local_details = self.apply_local_adjustments(
            scores,
            raw_dict,
            fs,
            deltas,
            rise_preds.get("1h"),
            drawdown_preds.get("1h"),
        )

        ic_periods = {
            "1h": self.ic_scores.get("1h", 1.0),
            "4h": self.ic_scores.get("4h", 1.0),
            "d1": self.ic_scores.get("d1", 1.0),
        }
        w1, w4, wd = self.get_ic_period_weights(ic_periods)

        fused_score, consensus_all, consensus_14, consensus_4d1 = self.fuse_multi_cycle(
            scores,
            (w1, w4, wd),
            local_details.get("strong_confirm_4h", False),
        )

        raw_1h = raw_dict.get("1h", {})
        if raw_1h.get("break_resistance_1h"):
            fused_score *= 1.12
            local_details["breakout_boost"] = 0.12

        logic_score = fused_score
        env_score = 1.0
        risk_score = 1.0

        if global_metrics is not None:
            dom = global_metrics.get("btc_dom_chg")
            if "btc_dominance" in global_metrics:
                self.btc_dom_history.append(global_metrics["btc_dominance"])
                if len(self.btc_dom_history) >= 5:
                    short = np.mean(list(self.btc_dom_history)[-5:])
                    long = np.mean(self.btc_dom_history)
                    dom_diff = (short - long) / long if long else 0
                    if dom is None:
                        dom = dom_diff
                    else:
                        dom += dom_diff
            if dom is not None:
                if symbol and str(symbol).upper().startswith("BTC"):
                    fused_score *= 1 + 0.1 * dom
                else:
                    fused_score *= 1 - 0.1 * dom

            eth_dom = global_metrics.get("eth_dom_chg")
            if "eth_dominance" in global_metrics:
                if not hasattr(self, "eth_dom_history"):
                    self.eth_dom_history = deque(maxlen=500)
                self.eth_dom_history.append(global_metrics["eth_dominance"])
                if len(self.eth_dom_history) >= 5:
                    short_e = np.mean(list(self.eth_dom_history)[-5:])
                    long_e = np.mean(self.eth_dom_history)
                    dom_diff_e = (short_e - long_e) / long_e if long_e else 0
                    if eth_dom is None:
                        eth_dom = dom_diff_e
                    else:
                        eth_dom += dom_diff_e
            if eth_dom is not None:
                if symbol and str(symbol).upper().startswith("ETH"):
                    fused_score *= 1 + 0.1 * eth_dom
                else:
                    fused_score *= 1 + 0.05 * eth_dom
            btc_mcap = global_metrics.get("btc_mcap_growth")
            alt_mcap = global_metrics.get("alt_mcap_growth")
            mcap_g = global_metrics.get("mcap_growth")
            if symbol and str(symbol).upper().startswith("BTC"):
                base_mcap = btc_mcap if btc_mcap is not None else mcap_g
            else:
                base_mcap = alt_mcap if alt_mcap is not None else mcap_g
            if base_mcap is not None:
                fused_score *= 1 + 0.1 * base_mcap
            vol_c = global_metrics.get("vol_chg")
            if vol_c is not None:
                fused_score *= 1 + 0.05 * vol_c
            hot = global_metrics.get("hot_sector_strength")
            if hot is not None:
                corr = global_metrics.get("sector_corr")
                if corr is None:
                    hot_name = global_metrics.get("hot_sector")
                    if hot_name and symbol:
                        cats = self.symbol_categories.get(str(symbol).upper())
                        if cats:
                            if isinstance(cats, str):
                                cats = [c.strip() for c in cats.split(',') if c.strip()]
                            corr = 1.0 if hot_name in cats else 0.0
                if corr is None:
                    corr = 1.0
                fused_score *= 1 + 0.05 * hot * corr

        env_score = fused_score / logic_score if logic_score != 0 else 1.0
        oi_overheat = False
        th_oi = None
        oi_chg = None
        if open_interest is not None:
            oi_chg = open_interest.get("oi_chg")
            if oi_chg is not None:
                with self._lock:
                    cache = self._get_symbol_cache(symbol)
                    cache["oi_change_history"].append(oi_chg)
                th_oi = self.get_dynamic_oi_threshold(pred_vol=vol_preds.get("1h"))
                fused_score, oi_overheat = self.apply_oi_overheat_protection(
                    fused_score, oi_chg, th_oi
                )

        mom5 = std_1h.get("mom_5m_roll1h")
        mom15 = std_1h.get("mom_15m_roll1h")
        mom_vals = [v for v in (mom5, mom15) if v is not None]
        short_mom = float(np.nanmean(mom_vals)) if mom_vals else 0.0
        if np.isnan(short_mom):
            short_mom = 0.0

        ob_imb = 0.0 if ob_imb is None or np.isnan(ob_imb) else float(ob_imb)

        if fused_score > 0:
            if short_mom > 0 and ob_imb > 0:
                fused_score *= 1.1
            elif short_mom < 0 or ob_imb < 0:
                fused_score *= 0.9
        elif fused_score < 0:
            if short_mom < 0 and ob_imb < 0:
                fused_score *= 1.1
            elif short_mom > 0 or ob_imb > 0:
                fused_score *= 0.9

        confirm_15m = 0.0
        if std_15m:
            rsi15 = std_15m.get("rsi_15m")
            ema15 = std_15m.get("ema_diff_15m")
            if rsi15 is not None:
                confirm_15m += (float(rsi15) - 50) / 50
            if ema15 is not None:
                confirm_15m += np.tanh(float(ema15) * 5)
            confirm_15m /= 2
            if fused_score > 0 and confirm_15m < -0.1:
                fused_score *= 0.85
            elif fused_score < 0 and confirm_15m > 0.1:
                fused_score *= 0.85

        result = {
            "fused_score": fused_score,
            "logic_score": logic_score,
            "env_score": env_score,
            "risk_score": risk_score,
            "fs": fs,
            "scores": scores,
            "local_details": local_details,
            "consensus_all": consensus_all,
            "consensus_14": consensus_14,
            "consensus_4d1": consensus_4d1,
            "short_mom": short_mom,
            "ob_imb": ob_imb,
            "confirm_15m": confirm_15m,
            "oi_overheat": oi_overheat,
            "th_oi": th_oi,
            "oi_chg": oi_chg,
        }
        self._cache_set(self._factor_score_cache, key, result)
        return result

    def apply_risk_filters(
        self,
        fused_score: float,
        logic_score: float,
        env_score: float,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        raw_f1h: dict,
        raw_f4h: dict,
        raw_fd1: dict,
        vol_preds: dict,
        open_interest: dict | None,
        all_scores_list: list | None,
        rev_dir: int,
        cache: dict,
        global_metrics: dict | None,
        features_1h: dict,
        features_4h: dict,
        features_d1: dict,
        symbol: str | None,
    ):
        """执行风险限制与拥挤度检查"""
        atr_1h = features_1h.get("atr_pct_1h", 0)
        adx_1h = features_1h.get("adx_1h", 0)
        funding_1h = features_1h.get("funding_rate_1h", 0) or 0

        atr_4h = features_4h.get("atr_pct_4h", 0) if features_4h else None
        adx_4h = features_4h.get("adx_4h", 0) if features_4h else None
        atr_d1 = features_d1.get("atr_pct_d1", 0) if features_d1 else None
        adx_d1 = features_d1.get("adx_d1", 0) if features_d1 else None

        vix_p = None
        if global_metrics is not None:
            vix_p = global_metrics.get("vix_proxy")
        if vix_p is None and open_interest is not None:
            vix_p = open_interest.get("vix_proxy")

        regime = self.detect_market_regime(adx_1h, adx_4h or 0, adx_d1 or 0)
        if std_d1.get("break_support_d1") == 1 and std_d1.get("rsi_d1", 50) < 30:
            regime = "range"
            rev_dir = 1
        cfg_th = self.signal_threshold_cfg
        base_th, rev_boost = self.dynamic_threshold(
            atr_1h,
            adx_1h,
            funding_1h,
            atr_4h=atr_4h,
            adx_4h=adx_4h,
            atr_d1=atr_d1,
            adx_d1=adx_d1,
            pred_vol=vol_preds.get("1h"),
            pred_vol_4h=vol_preds.get("4h"),
            pred_vol_d1=vol_preds.get("d1"),
            vix_proxy=vix_p,
            regime=regime,
            base=cfg_th.get("base_th", 0.08),
            reversal=bool(rev_dir),
            history_scores=cache["history_scores"],
        )
        if rev_dir != 0:
            fused_score += rev_boost * rev_dir
            self._cooldown = 0

        funding_conflicts = 0
        for p, raw_f in [("1h", raw_f1h), ("4h", raw_f4h), ("d1", raw_fd1)]:
            if raw_f is None:
                continue
            f_rate = raw_f.get(f"funding_rate_{p}", 0)
            if abs(f_rate) > 0.0005 and np.sign(f_rate) * np.sign(fused_score) < 0:
                penalty = min(abs(f_rate) * 20, 0.20)
                fused_score *= 1 - penalty
                funding_conflicts += 1
        if funding_conflicts >= 2:
            fused_score *= 0.85 ** funding_conflicts

        crowding_factor = 1.0
        if not cache.get("oi_overheat") and all_scores_list is not None:
            crowding_factor = self.crowding_protection(all_scores_list, fused_score, base_th)
            fused_score *= crowding_factor
        th_oi = cache.get("th_oi")
        oi_chg = cache.get("oi_chg")
        if th_oi is not None and oi_chg is not None:
            oi_crowd = abs(oi_chg) / max(th_oi, 1e-6)
            mult = 1 - min(0.5, oi_crowd * 0.5)
            if mult < 1:
                logging.debug(
                    "oi change %.4f threshold %.3f -> crowding mult %.3f for %s",
                    oi_chg,
                    th_oi,
                    mult,
                    symbol,
                )
                fused_score *= mult
                crowding_factor *= mult
        risk_score = self.risk_manager.fused_to_risk(
            fused_score,
            logic_score,
            env_score,
        )
        risk_score = min(1.0, risk_score)

        raw_score = logic_score * env_score * risk_score
        fused_score = raw_score - self.risk_adjust_factor * risk_score
        if abs(fused_score) < self.risk_adjust_threshold:
            return None

        if (
            risk_score > self.risk_score_limit
            or crowding_factor < 0
            or crowding_factor > self.crowding_limit
        ):
            return None

        with self._lock:
            cache["history_scores"].append(fused_score)
            self.all_scores_list.append(fused_score)

        return {
            "fused_score": fused_score,
            "risk_score": risk_score,
            "crowding_factor": crowding_factor,
            "base_th": base_th,
            "regime": regime,
            "rev_dir": rev_dir,
            "funding_conflicts": funding_conflicts,
        }

    def finalize_position(
        self,
        fused_score: float,
        risk_info: dict,
        ai_scores: dict,
        fs: dict,
        scores: dict,
        std_1h: dict,
        std_4h: dict,
        std_d1: dict,
        std_15m: dict,
        raw_f1h: dict,
        raw_f4h: dict,
        raw_fd1: dict,
        raw_f15m: dict,
        vol_preds: dict,
        rise_preds: dict,
        drawdown_preds: dict,
        short_mom: float,
        ob_imb: float,
        confirm_15m: float,
        oversold_reversal: bool,
        cache: dict,
        symbol: str | None,
    ):
        """根据阈值与信号方向计算仓位和止盈止损"""
        base_th = risk_info["base_th"]
        crowding_factor = risk_info["crowding_factor"]
        risk_score = risk_info["risk_score"]
        regime = risk_info["regime"]
        rev_dir = risk_info["rev_dir"]
        funding_conflicts = risk_info["funding_conflicts"]

        vol_ratio_1h_4h = std_1h.get("vol_ratio_1h_4h")
        if vol_ratio_1h_4h is None and std_4h is not None:
            vol_ratio_1h_4h = std_4h.get("vol_ratio_1h_4h")
        if vol_ratio_1h_4h is None:
            vol_ratio_1h_4h = 1.0
        ob_th = max(
            self.ob_th_params["min_ob_th"],
            self.ob_th_params["dynamic_factor"] * vol_ratio_1h_4h,
        )
        if ob_imb is not None and ob_imb > ob_th:
            ob_dir = 1
        elif ob_imb is not None and ob_imb < -ob_th:
            ob_dir = -1
        else:
            ob_dir = 0

        short_mom_dir = int(np.sign(short_mom)) if short_mom != 0 else 0
        fast_cross_dir = 0
        if raw_f15m:
            hist15 = cache.get("_raw_history", {}).get("15m", [])
            prev15 = hist15[-1] if hist15 else None
            rsi_c = raw_f15m.get("rsi_fast_15m")
            stoch_c = raw_f15m.get("stoch_fast_15m")
            if prev15 is not None:
                rsi_p = prev15.get("rsi_fast_15m")
                stoch_p = prev15.get("stoch_fast_15m")
                if rsi_p is not None and rsi_c is not None:
                    if rsi_p < 30 <= rsi_c:
                        fast_cross_dir += 1
                    elif rsi_p > 70 and rsi_c <= 70:
                        fast_cross_dir -= 1
                if stoch_p is not None and stoch_c is not None:
                    if stoch_p < 20 <= stoch_c:
                        fast_cross_dir += 1
                    elif stoch_p > 80 and stoch_c <= 80:
                        fast_cross_dir -= 1
            fast_cross_dir = int(np.sign(fast_cross_dir))
        vol_breakout_val = std_1h.get("vol_breakout_1h")
        vol_breakout_dir = 1 if vol_breakout_val and vol_breakout_val > 0 else 0

        trend_dir = int(np.sign(fs['1h'].get('trend', 0)))
        confirm_dir = int(np.sign(confirm_15m)) if confirm_15m else 0

        th = self.ai_dir_eps
        if ai_scores["1h"] >= th:
            ai_dir = 1
        elif ai_scores["1h"] <= -th:
            ai_dir = -1
        else:
            ai_dir = 0

        vw = self.vote_weights
        vote = (
            vw.get("ai", self.vote_params["weight_ai"]) * ai_dir
            + vw.get("mom", 1) * short_mom_dir
            + vw.get("vol_breakout", 1) * vol_breakout_dir
            + vw.get("trend", 1) * trend_dir
            + vw.get("confirm_15m", 1) * confirm_dir
        )
        conflict_filter_triggered = False
        if (
            std_1h.get("donchian_perc_1h", 0) > 0.7
            and (std_4h or {}).get("donchian_perc_4h", 1) < 0.2
        ):
            vote = 0
            conflict_filter_triggered = True
        strong_confirm_vote = abs(vote) >= self.vote_params["strong_min"]

        strong_min = self.vote_params["strong_min"]
        conf_vote = sigmoid_confidence(vote, strong_min, 1)
        if abs(vote) >= strong_min:
            fused_score *= max(1, conf_vote)

        vote_sign = int(np.sign(vote))
        if vote_sign != 0 and np.sign(fused_score) != vote_sign:
            strong_min = max(self.vote_params.get("strong_min", 1), 1)
            penalty = abs(vote) / strong_min
            fused_score *= 0.5 ** penalty

        rsi = raw_fd1.get("rsi_d1")
        adx = raw_fd1.get("adx_d1", 0)
        rebound_flag = False
        if rsi is not None and rsi < 30:
            hist = cache.get("_raw_history", {}).get("1h", [])
            rsi_hist = [r.get("rsi_1h") for r in hist]
            if raw_f1h.get("rsi_1h") is not None:
                rsi_hist.append(raw_f1h.get("rsi_1h"))
            price_seq = [r.get("close") for r in hist]
            if raw_f1h.get("close") is not None:
                price_seq.append(raw_f1h.get("close"))
            if (
                len(rsi_hist) >= 2
                and len(price_seq) >= 2
                and price_seq[-1] < price_seq[-2]
                and rsi_hist[-1] > rsi_hist[-2]
            ):
                rebound_flag = True
            hammer = raw_f1h.get("long_lower_shadow_1h", 0) > 0.6
            rebound_flag = rebound_flag or hammer
            if rebound_flag:
                fused_score += 0.3


        cfg_th_sig = self.signal_threshold_cfg
        grad_dir = sigmoid_dir(
            fused_score,
            base_th,
            cfg_th_sig.get("gamma", 0.9),
        )
        st1 = int(np.sign(std_1h.get("supertrend_dir_1h", 0)))
        st4 = int(np.sign(std_4h.get("supertrend_dir_4h", 0))) if std_4h else 0
        stdir = int(np.sign(std_d1.get("supertrend_dir_d1", 0))) if std_d1 else 0
        gd = int(np.sign(grad_dir))
        if all(v != 0 for v in (st1, st4, stdir)):
            if not (st1 == gd == st4 == stdir):
                return None
        direction = 0 if grad_dir == 0 else int(np.sign(grad_dir))

        if regime == "range":
            atr_v = (raw_f1h or std_1h).get("atr_pct_1h")
            bb_w = (raw_f1h or std_1h).get("bb_width_1h")
            low_vol = False
            if atr_v is not None and atr_v < 0.005:
                low_vol = True
            if bb_w is not None and bb_w < 0.01:
                low_vol = True
            if low_vol:
                direction = 0
            elif vol_breakout_val != 1 or conf_vote < 0.15:
                direction = 0

        if self._cooldown > 0:
            self._cooldown -= 1

        if self._last_signal != 0 and direction != 0 and direction != self._last_signal:
            flip_th = max(base_th, self.flip_coeff * abs(self._last_score))
            if abs(fused_score) < flip_th or self._cooldown > 0:
                direction = self._last_signal
            else:
                self._cooldown = 2

        prev_vote = getattr(self, "_prev_vote", 0)

        align_count = 0
        if direction != 0:
            for p in ("1h", "4h", "d1"):
                if np.sign(fs[p]["trend"]) == direction:
                    align_count += 1
            min_align = self.min_trend_align if regime == "trend" else max(
                self.min_trend_align - 1, 0
            )
            if align_count < min_align:
                direction = 0

        base_coeff = self.pos_coeff_range if regime == "range" else self.pos_coeff_trend
        confidence_factor = 1.0
        if risk_info.get("consensus_all"):
            confidence_factor += 0.1
        if strong_confirm_vote:
            confidence_factor += 0.05
        vol_ratio = std_1h.get("vol_ma_ratio_1h")
        tier = None
        fused_score = soft_clip(fused_score, k=1.0)
        pos_size, direction, tier, zero_reason = self.compute_position_size(
            grad_dir=grad_dir,
            base_coeff=base_coeff,
            confidence_factor=confidence_factor,
            vol_ratio=vol_ratio,
            fused_score=fused_score,
            base_th=base_th,
            regime=regime,
            oi_overheat=risk_info.get("oi_overheat", False),
            vol_p=vol_preds.get("1h"),
            risk_score=risk_score,
            crowding_factor=crowding_factor,
            cfg_th_sig=cfg_th_sig,
            scores=scores,
            direction=direction,
            exit_mult=(
                self.compute_exit_multiplier(vote, prev_vote, self._last_signal)
                if direction == self._last_signal and self._last_signal != 0
                else 1.0
            ),
            consensus_all=risk_info.get("consensus_all", False),
        )


        if risk_info.get("oi_overheat"):
            pos_size *= 0.5

        pos_map = base_th * 2.0
        if risk_score > 1 or risk_info.get("logic_score", 0) < -0.3:
            pos_map = min(pos_map, 0.5)
        pos_size = min(pos_size, pos_map)
        if conflict_filter_triggered:
            pos_size = 0.0
            zero_reason = zero_reason or "conflict_filter"

        price = (raw_f1h or std_1h).get("close", 0)
        if raw_f4h is not None and "atr_pct_4h" in raw_f4h:
            atr_pct_4h = raw_f4h["atr_pct_4h"]
        else:
            atr_pct_4h = (raw_f4h or std_4h).get("atr_pct_4h", 0)
        atr_raw = (raw_f1h or std_1h).get("atr_pct_1h", 0)
        atr_abs = np.hypot(atr_raw, atr_pct_4h) * price
        atr_abs = max(atr_abs, 0.005 * price)
        take_profit = stop_loss = None
        if direction != 0:
            take_profit, stop_loss = self.compute_tp_sl(
                price,
                atr_abs,
                direction,
                rise_pred=rise_preds.get("1h"),
                drawdown_pred=drawdown_preds.get("1h"),
                regime=regime,
            )

        rsi = raw_fd1.get("rsi_d1")
        adx = raw_fd1.get("adx_d1", 0)
        if direction == 1 and rsi is not None and rsi > 70:
            if adx < 25:
                pos_size *= 0.5
            else:
                pos_size *= 0.8
                if stop_loss is not None:
                    stop_loss *= 1.2
        elif direction == -1 and rsi is not None and rsi < 30:
            if adx < 25:
                pos_size *= 0.5
            else:
                pos_size *= 0.8
                if stop_loss is not None:
                    stop_loss *= 1.2
        if rebound_flag and direction == 1 and pos_size < tier * 0.2:
            pos_size = tier * 0.2
            zero_reason = None

        logic_score = risk_info.get("logic_score", 0.0)
        env_score = risk_info.get("env_score", 1.0)
        score_raw = logic_score * env_score * risk_score
        score_raw -= self.risk_adjust_factor * risk_score
        if vote_sign != 0 and np.sign(score_raw) != vote_sign:
            strong_min = max(self.vote_params.get("strong_min", 1), 1)
            penalty = abs(vote) / strong_min
            score_raw *= 0.5 ** penalty
        final_score = float(np.tanh(score_raw))

        with self._lock:
            self._last_signal = int(np.sign(direction)) if direction else 0
            self._last_score = fused_score
            self._prev_vote = vote
            cache["_prev_raw"]["15m"] = std_15m
            cache["_prev_raw"]["1h"] = std_1h
            cache["_prev_raw"]["4h"] = std_4h
            cache["_prev_raw"]["d1"] = std_d1
            for p, raw in [
                ("15m", raw_f15m),
                ("1h", raw_f1h),
                ("4h", raw_f4h),
                ("d1", raw_fd1),
            ]:
                maxlen = 4 if p in ("15m", "1h") else 2
                cache["_raw_history"].setdefault(p, deque(maxlen=maxlen)).append(raw)

        final_details = {
            "ai": {"1h": ai_scores["1h"], "4h": ai_scores["4h"], "d1": ai_scores["d1"]},
            "factors": {"1h": fs["1h"], "4h": fs["4h"], "d1": fs["d1"]},
            "scores": {"1h": scores["1h"], "4h": scores["4h"], "d1": scores["d1"]},
            "vote": {"value": vote, "confidence": conf_vote, "ob_th": ob_th},
            "protect": {
                "oi_overheat": risk_info.get("oi_overheat"),
                "oi_threshold": risk_info.get("th_oi"),
                "crowding_factor": crowding_factor,
                "funding_conflicts": funding_conflicts,
            },
            "env": {
                "logic_score": risk_info.get("logic_score", logic_score),
                "env_score": risk_info.get("env_score", env_score),
                "risk_score": risk_score,
            },
            "exit": {
                "regime": regime,
                "reversal_flag": rev_dir,
                "dynamic_th_final": base_th,
            },
            "grad_dir": float(grad_dir),
            "pos_size": float(pos_size),
            "short_momentum": short_mom,
            "ob_imbalance": ob_imb,
            "fast_cross": fast_cross_dir,
            "vol_ratio": vol_ratio,
            "position_tier": tier,
            "confidence_factor": confidence_factor,
            "consensus_all": risk_info.get("consensus_all"),
            "consensus_14": risk_info.get("consensus_14"),
            "consensus_4d1": risk_info.get("consensus_4d1"),
            "oversold_reversal": oversold_reversal,
            "conflict_filter_triggered": conflict_filter_triggered,
            "confirm_15m": confirm_15m,
        }
        final_details.update(risk_info.get("local_details", {}))

        return {
            "signal": int(direction),
            "score": final_score,
            "position_size": float(round(pos_size, 4)),
            "zero_reason": zero_reason if pos_size == 0 else None,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "details": final_details,
        }

    def generate_signal(
        self,
        features_1h,
        features_4h,
        features_d1,
        features_15m=None,
        all_scores_list=None,
        raw_features_1h=None,
        raw_features_4h=None,
        raw_features_d1=None,
        raw_features_15m=None,
        *,
        global_metrics=None,
        open_interest=None,
        order_book_imbalance=None,
        symbol=None,
    ):
        """
        输入：
            - features_1h: dict，当前 1h 周期特征（Robust Z 标准化）
            - features_4h: dict，当前 4h 周期特征（Robust Z 标准化）
            - features_d1: dict，当前 d1 周期特征（Robust Z 标准化）
            - features_15m: dict，可选，15分钟周期特征，用于确认方向
            - all_scores_list: list，可选，当前所有币种的 fused_score 列表，用于极端行情保护
            - raw_features_1h: dict，可选，未标准化的 1h 原始特征
            - raw_features_4h: dict，可选，未标准化的 4h 原始特征
            - raw_features_d1: dict，可选，未标准化的 d1 原始特征
            - raw_features_15m: dict，可选，未标准化的 15m 原始特征
            - order_book_imbalance: float，可选，L2 Order Book 的买卖盘差值比
            - symbol: str，可选，当前币种，如 'BTCUSDT'
        输出：
            一个 dict，包含 'signal'、'score'、'position_size'、'take_profit'、'stop_loss' 和 'details'

        features_* 为主要输入，多因子评分、动态阈值等逻辑均优先使用这些标准化后的特征。
        raw_features_* 仅在计算绝对价格或绝对幅度（例如止盈止损价）时作为补充使用。
        """

        (
            features_1h,
            features_4h,
            features_d1,
            features_15m,
            raw_features_1h,
            raw_features_4h,
            raw_features_d1,
            raw_features_15m,
        ) = self._normalize_inputs(
            features_1h,
            features_4h,
            features_d1,
            features_15m,
            raw_features_1h,
            raw_features_4h,
            raw_features_d1,
            raw_features_15m,
        )

        ob_imb = (
            order_book_imbalance
            if order_book_imbalance is not None
            else features_1h.get('bid_ask_imbalance')
        )

        std_1h = features_1h or {}
        std_4h = features_4h or {}
        std_d1 = features_d1 or {}
        std_15m = features_15m or {}
        raw_f1h = raw_features_1h or {}
        raw_f4h = raw_features_4h or {}
        raw_fd1 = raw_features_d1 or {}
        raw_f15m = raw_features_15m or {}
        raw_dict = {'15m': raw_f15m, '1h': raw_f1h, '4h': raw_f4h, 'd1': raw_fd1}

        ts = (
            raw_features_1h.get('ts')
            or raw_features_1h.get('timestamp')
            or raw_features_15m.get('ts')
            or raw_features_15m.get('timestamp')
            or features_1h.get('ts')
            or features_1h.get('timestamp')
            or features_15m.get('ts')
            or features_15m.get('timestamp')
        )

        cache = self._get_symbol_cache(symbol)
        with self._lock:
            hist_1h = cache["_raw_history"].get('1h', deque(maxlen=4))
        price_hist = [r.get('close') for r in hist_1h]
        price_hist.append((raw_features_1h or features_1h).get('close'))
        price_hist = [p for p in price_hist if p is not None][-4:]

        coin = str(symbol).upper() if symbol else ""
        rev_dir = self.detect_reversal(
            np.array(price_hist, dtype=float),
            (raw_features_1h or features_1h).get('atr_pct_1h'),
            (raw_features_1h or features_1h).get('vol_ma_ratio_1h'),
        )


        deltas = {}
        for p, feats, keys in [
            ("15m", std_15m, self.core_keys.get("15m", [])),
            ("1h", std_1h, self.core_keys["1h"]),
            ("4h", std_4h, self.core_keys["4h"]),
            ("d1", std_d1, self.core_keys["d1"]),
        ]:
            prev = cache["_prev_raw"].get(p)
            deltas[p] = self._calc_deltas(feats, prev, keys)

        # ===== 1. 计算 AI 部分的分数（映射到 [-1, 1]） =====
        ai_scores, vol_preds, rise_preds, drawdown_preds, oversold_reversal = self.compute_ai_scores(
            std_1h, std_4h, std_d1, raw_fd1
        )

        result = self.compute_factor_scores(
            ai_scores,
            std_1h,
            std_4h,
            std_d1,
            std_15m,
            raw_dict,
            deltas,
            rise_preds,
            drawdown_preds, vol_preds,
            global_metrics,
            open_interest,
            ob_imb,
            symbol,
        )
        fused_score = result["fused_score"]
        logic_score = result["logic_score"]
        env_score = result["env_score"]
        risk_score = result["risk_score"]
        fs = result["fs"]
        scores = result["scores"]
        local_details = result["local_details"]
        consensus_all = result["consensus_all"]
        consensus_14 = result["consensus_14"]
        consensus_4d1 = result["consensus_4d1"]
        short_mom = result["short_mom"]
        ob_imb = result["ob_imb"]
        confirm_15m = result["confirm_15m"]
        oi_overheat = result["oi_overheat"]
        th_oi = result["th_oi"]
        oi_chg = result["oi_chg"]

        # ===== 7. 如果 fused_score 为 NaN，直接返回无信号 =====
        if fused_score is None or (isinstance(fused_score, float) and np.isnan(fused_score)):
            logging.debug("Fused score NaN, returning 0 signal")
            self._last_score = fused_score
            with self._lock:
                self._prev_raw["15m"] = std_15m
                self._prev_raw["1h"] = std_1h
                self._prev_raw["4h"] = std_4h
                self._prev_raw["d1"] = std_d1
                for p, raw in [("15m", raw_f15m), ("1h", raw_f1h), ("4h", raw_f4h), ("d1", raw_fd1)]:
                    maxlen = 4 if p in ("15m", "1h") else 2
                    self._raw_history.setdefault(p, deque(maxlen=maxlen)).append(raw)
            self._last_signal = 0
            self._cooldown = 0
            return {
                'signal': 0,
                'score': float('nan'),
                'position_size': 0.0,
                'take_profit': None,
                'stop_loss': None,
                'details': {
                    'ai_1h': ai_scores['1h'],   'ai_4h': ai_scores['4h'],   'ai_d1': ai_scores['d1'],
                    'factors_1h': fs['1h'],     'factors_4h': fs['4h'],     'factors_d1': fs['d1'],
                    'score_1h': scores['1h'],   'score_4h': scores['4h'],   'score_d1': scores['d1'],
                    'strong_confirm_4h': local_details.get('strong_confirm_4h'),
                    'consensus_14': consensus_14, 'consensus_all': consensus_all,
                    'vol_pred_1h': vol_preds.get('1h'),
                    'vol_pred_4h': vol_preds.get('4h'),
                    'vol_pred_d1': vol_preds.get('d1'),
                    'rise_pred_1h': rise_preds.get('1h'),
                    'rise_pred_4h': rise_preds.get('4h'),
                    'rise_pred_d1': rise_preds.get('d1'),
                    'drawdown_pred_1h': drawdown_preds.get('1h'),
                    'drawdown_pred_4h': drawdown_preds.get('4h'),
                    'drawdown_pred_d1': drawdown_preds.get('d1'),
                    'funding_conflicts': 0,
                    'confirm_15m': confirm_15m,
                    'note': 'fused_score was NaN'
                }
            }

# ===== 7b. 计算动态阈值 =====
        atr_1h = features_1h.get('atr_pct_1h', 0)
        adx_1h = features_1h.get('adx_1h', 0)
        funding_1h = features_1h.get('funding_rate_1h', 0) or 0

        atr_4h = features_4h.get('atr_pct_4h', 0) if features_4h else None
        adx_4h = features_4h.get('adx_4h', 0) if features_4h else None
        atr_d1 = features_d1.get('atr_pct_d1', 0) if features_d1 else None
        adx_d1 = features_d1.get('adx_d1', 0) if features_d1 else None

        vix_p = None
        if global_metrics is not None:
            vix_p = global_metrics.get('vix_proxy')
        if vix_p is None and open_interest is not None:
            vix_p = open_interest.get('vix_proxy')

        regime = self.detect_market_regime(adx_1h, adx_4h or 0, adx_d1 or 0)
        if std_d1.get('break_support_d1') == 1 and std_d1.get('rsi_d1', 50) < 30:
            regime = 'range'
            rev_dir = 1
        risk_info = self.apply_risk_filters(
            fused_score,
            logic_score,
            env_score,
            std_1h,
            std_4h,
            std_d1,
            raw_f1h,
            raw_f4h,
            raw_fd1,
            vol_preds,
            open_interest,
            all_scores_list,
            rev_dir,
            {
                "oi_overheat": oi_overheat,
                "th_oi": th_oi,
                "oi_chg": oi_chg,
                "history_scores": cache["history_scores"],
            },
            global_metrics,
            features_1h,
            features_4h,
            features_d1,
            symbol,
        )
        if risk_info is None:
            return None
        fused_score = risk_info["fused_score"]
        risk_score = risk_info["risk_score"]
        base_th = risk_info["base_th"]
        crowding_factor = risk_info["crowding_factor"]
        regime = risk_info["regime"]
        rev_dir = risk_info["rev_dir"]
        funding_conflicts = risk_info["funding_conflicts"]
        risk_info["logic_score"] = logic_score
        risk_info["env_score"] = env_score
        risk_info["consensus_all"] = consensus_all
        risk_info["consensus_14"] = consensus_14
        risk_info["consensus_4d1"] = consensus_4d1
        risk_info["local_details"] = local_details
        result = self.finalize_position(
            fused_score,
            risk_info,
            ai_scores,
            fs,
            scores,
            std_1h,
            std_4h,
            std_d1,
            std_15m,
            raw_f1h,
            raw_f4h,
            raw_fd1,
            raw_f15m,
            vol_preds,
            rise_preds,
            drawdown_preds,
            short_mom,
            ob_imb,
            confirm_15m,
            oversold_reversal,
            cache,
            symbol,
        )
        if result is None:
            logger.debug(
                "step=%s fused=%.3f th=%.3f position skipped",
                ts,
                fused_score,
                base_th,
            )
            return None
        if all_scores_list is None:
            all_scores_list = self.all_scores_list
        logger.debug(
            "step=%s fused=%.3f th=%.3f pos=%.4f",
            ts,
            fused_score,
            base_th,
            result.get("position_size", 0.0),
        )
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# -*- coding: utf-8 -*-
import joblib
import numpy as np
import pandas as pd
from collections import Counter, deque
from pathlib import Path
import yaml
import threading
import logging
import time


logger = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)
# Set module logger to WARNING by default so importing modules can
# configure their own verbosity without receiving this module's INFO logs.
logger.setLevel(logging.WARNING)
# 默认配置路径
CONFIG_PATH = Path(__file__).resolve().parent / "utils" / "config.yaml"


# 退出信号滞后 bar 数默认值
EXIT_LAG_BARS_DEFAULT = 1

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


def softmax(x):
    """简单 softmax 实现"""
    arr = np.array(x, dtype=float)
    ex = np.exp(arr - np.nanmax(arr))
    return ex / ex.sum()


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
    ratio_low: float = 0.6,
    ratio_high: float = 2.0,
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
        return score * 1.05
    if ratio >= extreme_ratio or roc >= extreme_roc:
        return score * over
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
    denom = max(abs(logic_score) * max(abs(env_score), 1e-6), 1e-6)
    risk = abs(fused_score) / denom
    return min(risk, cap)


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
        ],
        "4h": ["rsi_4h", "macd_hist_4h", "ema_diff_4h"],
        "d1": ["rsi_d1", "macd_hist_d1", "ema_diff_d1"],
    }

    DELTA_PARAMS = {
        "rsi": (5, 1.0, 0.05),
        "macd_hist": (0.002, 100.0, 0.05),
        "ema_diff": (0.001, 100.0, 0.03),
        "atr_pct": (0.002, 100.0, 0.03),
        "vol_ma_ratio": (0.2, 1.0, 0.03),
        "funding_rate": (0.0005, 10000, 0.03),
    }

    VOTE_PARAMS = {
        "weight_ai": 3,
        "strong_min": 3,
        "conf_min": 0.30,
    }

    def __init__(
        self,
        model_paths,
        *,
        feature_cols_1h,
        feature_cols_4h,
        feature_cols_d1,
        history_window=220,
        symbol_categories=None,
        config_path=CONFIG_PATH,
        core_keys=None,
        delta_params=None,
        min_weight_ratio=0.3,
        th_window=150,
        th_decay=1.0,
    ):
        # 加载AI模型，同时保留训练时的 features 列名
        self.models = {}
        self.calibrators = {}
        for period, path_dict in model_paths.items():
            self.models[period] = {}
            self.calibrators[period] = {}
            for direction, path in path_dict.items():
                loaded = joblib.load(path)
                pipe = loaded["pipeline"]
                if hasattr(pipe, "set_output"):
                    pipe.set_output(transform="pandas")
                self.models[period][direction] = {
                    "pipeline": loaded["pipeline"],
                    "features": loaded["features"],
                }
                self.calibrators[period][direction] = loaded.get("calibrator")

        # 保存各时间周期对应的特征列（但后面不再直接用它；实际推理要以 loaded["features"] 为准）
        self.feature_cols_1h = feature_cols_1h
        self.feature_cols_4h = feature_cols_4h
        self.feature_cols_d1 = feature_cols_d1

        cfg = {}
        path = Path(config_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        self.cfg = cfg
        self.signal_threshold_cfg = cfg.get("signal_threshold", {})
        if "low_base" not in self.signal_threshold_cfg:
            self.signal_threshold_cfg["low_base"] = DEFAULT_LOW_BASE
        db_cfg = cfg.get("delta_boost", {})
        self.core_keys = core_keys or db_cfg.get("core_keys", self.DEFAULT_CORE_KEYS)
        self.delta_params = delta_params or db_cfg.get("params", self.DELTA_PARAMS)
        vote_cfg = cfg.get("vote_system", {})
        self.vote_params = {
            "weight_ai": vote_cfg.get("weight_ai", self.VOTE_PARAMS["weight_ai"]),
            "strong_min": vote_cfg.get("strong_min", self.VOTE_PARAMS["strong_min"]),
            "conf_min": vote_cfg.get("conf_min", self.VOTE_PARAMS["conf_min"]),
        }
        self.ai_dir_eps = vote_cfg.get("ai_dir_eps", DEFAULT_AI_DIR_EPS)
        self.vote_weights = cfg.get(
            "vote_weights",
            {"ob": 4, "short_mom": 2, "ai": self.vote_params["weight_ai"], "vol_breakout": 1},
        )

        filters_cfg = cfg.get("signal_filters", {})
        self.signal_filters = {
            "min_vote": filters_cfg.get("min_vote", 5),
            "confidence_vote": filters_cfg.get("confidence_vote", 0.2),
        }

        pc_cfg = cfg.get("position_coeff", {})
        self.pos_coeff_range = pc_cfg.get("range", DEFAULT_POS_K_RANGE)
        self.pos_coeff_trend = pc_cfg.get("trend", DEFAULT_POS_K_TREND)

        self.sentiment_alpha = cfg.get("sentiment_alpha", 0.5)
        self.cap_positive_scale = cfg.get("cap_positive_scale", 0.7)
        vg_cfg = cfg.get("volume_guard", {})
        self.volume_guard_params = {
            "weak": vg_cfg.get("weak_scale", 0.85),
            "over": vg_cfg.get("over_scale", 0.9),
            "ratio_low": vg_cfg.get("ratio_low", 0.6),
            "ratio_high": vg_cfg.get("ratio_high", 2.0),
            "roc_low": vg_cfg.get("roc_low", -20),
            "roc_high": vg_cfg.get("roc_high", 100),
        }
        ob_cfg = cfg.get("ob_threshold", {})
        self.ob_th_params = {
            "min_ob_th": ob_cfg.get("min_ob_th", 0.15),
            "dynamic_factor": ob_cfg.get("dynamic_factor", 0.08),
        }
        self.risk_score_cap = cfg.get("risk_score_cap", 5.0)
        self.exit_lag_bars = cfg.get("exit_lag_bars", EXIT_LAG_BARS_DEFAULT)
        oi_cfg = cfg.get("oi_protection", {})
        self.oi_scale = oi_cfg.get("scale", 0.8)
        self.max_same_direction_rate = oi_cfg.get("crowding_threshold", 0.90)
        self.veto_level = cfg.get("veto_level", 0.7)
        self.flip_coeff = cfg.get("flip_coeff", 0.5)
        self.th_down_d1 = self.cfg.get("th_down_d1", 0.74)
        self.min_weight_ratio = min_weight_ratio
        self.th_window = th_window
        self.th_decay = th_decay

        # 静态因子权重（后续可由动态IC接口进行更新）
        _base_weights = {
            'ai': 0.10374469306511298,
            'trend': 0.29541645926817756,
            'momentum': 0.2889591044255588,
            'volatility': 0.2,
            'volume': 0.1,
            'sentiment': 0.05,
            'funding': 0.05,
        }
        total_w = sum(_base_weights.values())
        if total_w <= 0:
            total_w = 1.0
        self.base_weights = {k: v / total_w for k, v in _base_weights.items()}

        # 当前权重，初始与 base_weights 相同
        self.current_weights = self.base_weights.copy()

        # 多线程访问历史数据时的互斥锁
        # 使用 RLock 以便在部分函数中嵌套调用
        self._lock = threading.RLock()

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


        # 当多个信号方向过于集中时，用于滤除极端行情（最大同向信号比例阈值）
        # 值由配置 oi_protection.crowding_threshold 控制

        # 保存上一次生成的信号，供滞后阈值逻辑使用
        self._last_signal = 0

        # 缓存上一周期的原始特征，便于计算指标变化量
        self._prev_raw = {p: None for p in ("1h", "4h", "d1")}

        # 定时更新因子权重
        self.start_weight_update_thread()

    def __getattr__(self, name):
        if name in {"eth_dom_history", "btc_dom_history", "oi_change_history", "history_scores"}:
            val = deque(maxlen=3000)
            setattr(self, name, val)
            return val
        if name == "ic_history":
            val = {k: deque(maxlen=3000) for k in self.base_weights}
            setattr(self, name, val)
            return val
        if name == "_lock":
            val = threading.RLock()
            setattr(self, name, val)
            return val
        if name == "_equity_drawdown":
            setattr(self, name, 0.0)
            return 0.0
        if name == "_last_score":
            setattr(self, name, 0.0)
            return 0.0
        if name == "_last_signal":
            setattr(self, name, 0)
            return 0
        if name == "_prev_vote":
            setattr(self, name, 0)
            return 0
        if name == "_exit_lag":
            setattr(self, name, 0)
            return 0
        if name == "_prev_raw":
            val = {p: None for p in ("1h", "4h", "d1")}
            setattr(self, name, val)
            return val
        if name == "core_keys":
            setattr(self, name, self.DEFAULT_CORE_KEYS.copy())
            return getattr(self, name)
        if name == "delta_params":
            setattr(self, name, self.DELTA_PARAMS.copy())
            return getattr(self, name)
        if name == "vote_params":
            setattr(self, name, self.VOTE_PARAMS.copy())
            return getattr(self, name)
        if name == "vote_weights":
            val = {"ob": 4, "short_mom": 2, "ai": 3, "vol_breakout": 1}
            setattr(self, name, val)
            return val
        if name == "min_weight_ratio":
            setattr(self, name, 0.3)
            return 0.3
        if name == "_cooldown":
            setattr(self, name, 0)
            return 0
        if name == "_volume_checked":
            setattr(self, name, False)
            return False
        if name == "_raw_history":
            val = {
                "1h": deque(maxlen=4),
                "4h": deque(maxlen=2),
                "d1": deque(maxlen=2),
            }
            setattr(self, name, val)
            return val
        if name == "symbol_data":
            val = {}
            setattr(self, name, val)
            return val
        if name == "calibrators":
            val = {p: {"up": None, "down": None} for p in ("1h", "4h", "d1")}
            setattr(self, name, val)
            return val
        if name == "ic_scores":
            val = {k: 1 for k in self.base_weights}
            setattr(self, name, val)
            return val
        if name == "exit_lag_bars":
            setattr(self, name, EXIT_LAG_BARS_DEFAULT)
            return EXIT_LAG_BARS_DEFAULT
        if name == "oi_scale":
            setattr(self, name, 0.8)
            return 0.8
        if name == "max_same_direction_rate":
            setattr(self, name, 0.85)
            return 0.85
        if name == "veto_level":
            setattr(self, name, 0.7)
            return 0.7
        if name == "flip_coeff":
            setattr(self, name, 0.5)
            return 0.5
        if name == "cfg":
            val = {}
            setattr(self, name, val)
            return val
        if name == "th_down_d1":
            setattr(self, name, 0.74)
            return 0.74
        if name == "signal_threshold_cfg":
            val = {
                "base_th": 0.08,
                "gamma": 0.05,
                "min_pos": 0.10,
                "low_base": DEFAULT_LOW_BASE,
                "rev_boost": 0.30,
                "rev_th_mult": 0.60,
            }
            setattr(self, name, val)
            return val
        if name == "ai_dir_eps":
            val = self.cfg.get("vote_system", {}).get("ai_dir_eps", DEFAULT_AI_DIR_EPS)
            setattr(self, name, val)
            return val
        if name == "pos_coeff_range":
            setattr(self, name, DEFAULT_POS_K_RANGE)
            return DEFAULT_POS_K_RANGE
        if name == "pos_coeff_trend":
            setattr(self, name, DEFAULT_POS_K_TREND)
            return DEFAULT_POS_K_TREND
        if name == "th_window":
            setattr(self, name, 150)
            return 150
        if name == "th_decay":
            setattr(self, name, 1.0)
            return 1.0
        raise AttributeError(name)


    def get_dynamic_oi_threshold(self, pred_vol=None, base=0.5, quantile=0.9):
        """根据历史 OI 变化率及预测波动率自适应阈值"""
        with self._lock:
            history = list(self.oi_change_history)
        if history:
            base = np.quantile(np.abs(history), 0.8)
        else:
            base = 0.2
        th = base
        if len(history) > 30:
            th = float(np.quantile(np.abs(history), quantile))
        if pred_vol is not None:
            th += min(0.1, abs(pred_vol) * 0.5)
        return max(th, 0.30)

    def detect_market_regime(self, adx1, adx4, adxd):
        """简易市场状态判别：根据平均ADX判断震荡或趋势"""
        avg_adx = np.nanmean([adx1, adx4, adxd])
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
    ):
        """计算止盈止损价格，可根据模型预测值微调"""
        if direction == 0:
            return None, None
        if price is None or price <= 0:
            return None, None  # 价格异常直接放弃
        if atr is None or atr == 0:
            atr = 0.005 * price

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
        else:
            if direction == 1:
                take_profit = price + tp_mult * atr
                stop_loss = price - sl_mult * atr
            else:
                take_profit = price - tp_mult * atr
                stop_loss = price + sl_mult * atr

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
                        "1h": deque(maxlen=4),
                        "4h": deque(maxlen=2),
                        "d1": deque(maxlen=2),
                    },
                    "_prev_raw": {p: None for p in ("1h", "4h", "d1")},
                }
            return self.symbol_data[symbol]

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
        if last_signal == 1:
            if prev_vote > 4 and 1 <= vote <= 4:
                exit_mult = 0.5
                exit_lag = 0
            elif vote <= 0:
                exit_lag += 1
                exit_mult = 0.0 if exit_lag >= self.exit_lag_bars else 0.5
            else:
                exit_lag = 0
        elif last_signal == -1:
            if prev_vote < -4 and -4 <= vote <= -1:
                exit_mult = 0.5
                exit_lag = 0
            elif vote >= 0:
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
    ) -> tuple[float, int]:
        """Calculate final position size given direction and risk factors."""

        tier = 0.1 + base_coeff * abs(grad_dir)
        pos_size = tier * confidence_factor * exit_mult
        pos_size *= crowding_factor
        if direction == 0:
            pos_size = 0.0

        if (
            regime == "range"
            and vol_ratio is not None
            and vol_ratio < 0.3
            and abs(fused_score) < base_th + 0.02
        ):
            direction = 0
            pos_size = 0.0


        if vol_p is not None:
            pos_size *= max(0.4, 1 - min(0.6, vol_p))

        if abs(risk_score) > 3 and confidence_factor > 1.05:
            pos_size = min(pos_size * 1.2, 1.0)

        if pos_size < cfg_th_sig.get("min_pos", 0.1):
            direction, pos_size = 0, 0.0

        if direction == 1 and scores.get("4h", 0) < -self.veto_level:
            direction, pos_size = 0, 0.0
        elif direction == -1 and scores.get("4h", 0) > self.veto_level:
            direction, pos_size = 0, 0.0

        return pos_size, direction

    # >>>>> 修改：改写 get_ai_score，让它自动从 self.models[...]["features"] 中取“训练时列名”
    def get_ai_score(self, features, model_up, model_down, calibrator_up=None, calibrator_down=None):
        """根据上下两个方向模型的概率输出计算 AI 得分"""

        def _build_df(model_dict):
            cols = model_dict["features"]
            row = {}
            missing = []
            for c in cols:
                val = features.get(c, np.nan)
                row[c] = [val]
                if c not in features:
                    missing.append(c)
            if missing:
                logging.debug("get_ai_score missing columns: %s", missing)
            df = pd.DataFrame(row)
            df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
            return df, missing

        X_up, missing_up = _build_df(model_up)
        X_down, missing_down = _build_df(model_down)
        if len(missing_up) > 3 or len(missing_down) > 3:
            return 0.0
        prob_up = model_up["pipeline"].predict_proba(X_up)[:, 1]
        prob_down = model_down["pipeline"].predict_proba(X_down)[:, 1]
        if calibrator_up is not None:
            prob_up = calibrator_up.transform(prob_up.reshape(-1, 1)).ravel()
        if calibrator_down is not None:
            prob_down = calibrator_down.transform(prob_down.reshape(-1, 1)).ravel()
        denom = prob_up + prob_down
        ai_score = np.where(denom == 0, 0.0, (prob_up - prob_down) / denom)
        ai_score = np.clip(ai_score, -1.0, 1.0)
        if ai_score.size == 1:
            return float(ai_score[0])
        return ai_score

    def get_ai_score_cls(self, features, model_dict):
        """从单个 cls 模型计算上涨/下跌概率差值"""
        cols = model_dict["features"]
        row = {c: [features.get(c, np.nan)] for c in cols}
        df = pd.DataFrame(row)
        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)

        probs = model_dict["pipeline"].predict_proba(df)[0]
        classes = getattr(model_dict["pipeline"], "classes_", np.arange(len(probs)))

        if len(classes) >= 3:
            idx_down = int(np.argmin(classes))
            idx_up = int(np.argmax(classes))
        else:
            idx_down = 0
            idx_up = min(1, len(probs) - 1)

        prob_down = probs[idx_down]
        prob_up = probs[idx_up]
        denom = prob_up + prob_down
        if denom == 0:
            return 0.0
        ai_score = (prob_up - prob_down) / denom
        return float(np.clip(ai_score, -1.0, 1.0))

    def get_vol_prediction(self, features, model_dict):
        """根据回归模型预测未来波动率"""
        lgb_model = model_dict["pipeline"]
        train_cols = model_dict["features"]

        row_data = {col: [features.get(col, 0)] for col in train_cols}
        X_df = pd.DataFrame(row_data)
        X_df = X_df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return float(lgb_model.predict(X_df)[0])

    def get_reg_prediction(self, features, model_dict):
        """通用回归模型预测"""
        model = model_dict["pipeline"]
        cols = model_dict["features"]
        row_data = {c: [features.get(c, 0)] for c in cols}
        df = pd.DataFrame(row_data)
        df = df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return float(model.predict(df)[0])

    # robust_signal_generator.py

    def get_factor_scores(self, features: dict, period: str) -> dict:
        """
        输入：
          - features: 单周期特征字典（如 {'ema_diff_1h': 0.12, 'boll_perc_1h': 0.45, ...}）
          - period:   "1h" / "4h" / "d1"
        输出：一个 dict，包含6个子因子得分。
        """

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
        )

        sentiment_raw = (
            (safe('fg_index_d1', 50) - 50) / 50
            + np.tanh(safe(f'btc_correlation_1h_{period}', 0))
            + np.tanh(safe(f'eth_correlation_1h_{period}', 0))
            + np.tanh(safe(f'price_diff_cg_{period}', 0) * 5)
            + np.tanh(safe(f'cg_market_cap_roc_{period}', 0) * 5)
            + np.tanh((safe(f'volume_cg_ratio_{period}', 1) - 1) * 2)
        )

        f_rate = safe(f'funding_rate_{period}', 0)
        f_anom = safe(f'funding_rate_anom_{period}', 0)
        thr = 0.0005  # 約 0.05% 年化
        if abs(f_rate) > thr:
            funding_raw = -np.tanh(f_rate * 4000)  # 4000 ≈ 1/0.00025，讓 ±0.002 ≈ tanh(8)
        else:
            funding_raw = np.tanh(f_rate * 4000)
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

        from param_search import compute_ic_scores

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
        base=0.12,
        regime=None,
        low_base=None,
        reversal=False,
        history_scores=None,
    ):
        """根据历史波动、趋势强度和市场情绪动态计算阈值并返回额外的反转加成"""

        import numpy as np

        th = float(base)

        # === 波动因子：max 避免双计 ===
        vol_th = 0.0
        main_vol = max(abs(atr), abs(pred_vol or 0.0))
        vol_th += min(0.10, main_vol * 4)
        if atr_4h is not None or pred_vol_4h is not None:
            vol_th += 0.5 * min(
                0.06, max(abs(atr_4h or 0), abs(pred_vol_4h or 0)) * 3
            )
        if atr_d1 is not None or pred_vol_d1 is not None:
            vol_th += 0.25 * min(
                0.06, max(abs(atr_d1 or 0), abs(pred_vol_d1 or 0)) * 3
            )
        th += vol_th * 1.5

        # === 趋势强度 ===
        th += min(0.12, max(adx - 25, 0) * 0.005)
        if adx_4h is not None:
            th += 0.5 * min(0.12, max(adx_4h - 25, 0) * 0.005)
        if adx_d1 is not None:
            th += 0.25 * min(0.12, max(adx_d1 - 25, 0) * 0.005)

        # === 资金费率 / 恐慌指数 ===
        th += min(0.08, abs(funding) * 8)
        if vix_proxy is not None:
            th += min(0.08, max(vix_proxy, 0.0) * 0.08)

        # === 历史 80 分位兜底 ===
        if history_scores is None:
            with self._lock:
                hist_scores = list(self.history_scores)
        else:
            hist_scores = list(history_scores)
        if len(hist_scores) > 100:
            scores = hist_scores
            if self.th_window:
                scores = scores[-int(self.th_window):]
            arr = np.abs(np.asarray(scores, dtype=float))
            if arr.size:
                if self.th_decay < 1.0:
                    weights = self.th_decay ** np.arange(len(arr))[::-1]
                    sorter = np.argsort(arr)
                    arr_sorted = arr[sorter]
                    w_sorted = weights[sorter]
                    cumsum = np.cumsum(w_sorted)
                    target = 0.80 * cumsum[-1]
                    q = np.interp(target, cumsum, arr_sorted)
                else:
                    q = np.quantile(arr, 0.80)
                th = max(th, q)

        # === 市场状态微调 ===
        if regime == "trend":
            th += 0.015
        elif regime == "range":
            th -= 0.020

        rev_boost = self.signal_threshold_cfg.get('rev_boost', 0.30)

        if reversal:
            th *= self.signal_threshold_cfg.get('rev_th_mult', 0.60)

        if low_base is None:
            low_base = self.signal_threshold_cfg.get('low_base', 0.10)

        return max(th, low_base), rev_boost

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

    def consensus_check(self, s1, s2, s3, min_agree=2):
        # 多周期方向共振（如调研建议），可加全分歧减弱等逻辑
        signs = np.sign([s1, s2, s3])
        non_zero = [g for g in signs if g != 0]
        if len(non_zero) < min_agree:
            return 0  # 无方向共振
        cnt = Counter(non_zero)
        if cnt.most_common(1)[0][1] >= min_agree:
            return int(cnt.most_common(1)[0][0])  # 返回方向
        return 0

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

        factor = 1.0 - 0.5 * intensity
        dd = getattr(self, "_equity_drawdown", 0.0)
        factor *= max(0.6, 1 - dd)
        return factor

    def apply_oi_overheat_protection(self, fused_score, oi_chg, th_oi):
        """根据 OI 变化率奖励或惩罚分数"""
        oi_overheat = False
        if th_oi is None:
            return fused_score, oi_overheat
        if abs(oi_chg) < th_oi:
            fused_score *= 1 + 0.08 * oi_chg
        else:
            logging.info("OI overheat detected: %.4f", oi_chg)
        fused_score *= self.oi_scale
        oi_overheat = True
        return fused_score, oi_overheat

    # ===== 新增辅助函数 =====
    def calc_factor_scores(self, ai_scores: dict, factor_scores: dict, weights: dict) -> dict:
        """计算未调整的各周期得分"""
        scores = {
            '1h': self.combine_score(ai_scores['1h'], factor_scores['1h'], weights),
            '4h': self.combine_score(ai_scores['4h'], factor_scores['4h'], weights),
            'd1': self.combine_score(ai_scores['d1'], factor_scores['d1'], weights),
        }
        logger.debug("factor scores: %s", scores)
        return scores

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
        elif consensus_14:
            total = w1 + w4
            fused = (w1 / total) * s1 + (w4 / total) * s4
            conf = 0.8
            if strong_confirm_4h:
                fused *= 1.10
        elif consensus_4d1:
            total = w4 + wd
            fused = (w4 / total) * s4 + (wd / total) * sd
            conf = 0.7
        else:
            fused = s1
            conf = 0.6

        fused_score = fused * conf
        logger.debug(
            "fuse scores s1=%.3f s4=%.3f sd=%.3f -> %.3f",
            s1,
            s4,
            sd,
            fused_score,
        )
        return fused_score, consensus_all, consensus_14, consensus_4d1

    def generate_signal(
        self,
        features_1h,
        features_4h,
        features_d1,
        all_scores_list=None,
        raw_features_1h=None,
        raw_features_4h=None,
        raw_features_d1=None,
        *,
        global_metrics=None,
        open_interest=None,
        order_book_imbalance=None,
        symbol=None,
    ):
        """
        输入：
            - features_1h: dict，当前 1h 周期下的全部特征键值对（已标准化）
            - features_4h: dict，当前 4h 周期下的全部特征键值对（已标准化）
            - features_d1: dict，当前 d1 周期下的全部特征键值对（已标准化）
            - all_scores_list: list，可选，当前所有币种的 fused_score 列表，用于极端行情保护
            - raw_features_1h: dict，可选，未标准化的 1h 特征
            - raw_features_4h: dict，可选，未标准化的 4h 特征；其中 atr_pct_4h 为实际
              比例（如 0.05 表示 5%），在计算止盈止损和指标计算时会优先使用
            - raw_features_d1: dict，可选，未标准化的 d1 特征
            - order_book_imbalance: float，可选，L2 Order Book 的买卖盘差值比
            - symbol: str，可选，当前币种，如 'BTCUSDT'
        输出：
            一个 dict，包含 'signal'、'score'、'position_size'、'take_profit'、'stop_loss' 和 'details'
        说明：若传入 raw_features_*，则多因子计算与动态阈值、止盈止损均使用原始数据，
              标准化后的 features_* 仅用于 AI 模型预测。
        """

        ob_imb = (
            order_book_imbalance
            if order_book_imbalance is not None
            else (raw_features_1h or features_1h).get('bid_ask_imbalance')
        )

        std_1h = features_1h
        std_4h = features_4h
        std_d1 = features_d1
        raw_f1h = {**features_1h, **(raw_features_1h or {})}
        raw_f4h = {**features_4h, **(raw_features_4h or {})}
        raw_fd1 = {**features_d1, **(raw_features_d1 or {})}

        cache = self._get_symbol_cache(symbol)
        with self._lock:
            hist_1h = cache["_raw_history"].get('1h', deque(maxlen=4))
        price_hist = [r.get('close') for r in hist_1h]
        price_hist.append(raw_f1h.get('close'))
        price_hist = [p for p in price_hist if p is not None][-4:]

        coin = str(symbol).upper() if symbol else ""
        rev_dir = self.detect_reversal(
            np.array(price_hist, dtype=float),
            raw_f1h.get('atr_pct_1h'),
            raw_f1h.get('vol_ma_ratio_1h'),
        )


        deltas = {}
        for p, raw, keys in [
            ("1h", raw_f1h, self.core_keys["1h"]),
            ("4h", raw_f4h, self.core_keys["4h"]),
            ("d1", raw_fd1, self.core_keys["d1"]),
        ]:
            maxlen = 4 if p == "1h" else 2
            hist = cache["_raw_history"].get(p, deque(maxlen=maxlen))
            prev = hist[0] if len(hist) == 2 else cache["_prev_raw"].get(p)
            deltas[p] = self._calc_deltas(raw, prev, keys)

        # ===== 1. 计算 AI 部分的分数（映射到 [-1, 1]） =====
        ai_scores = {}
        vol_preds = {}
        rise_preds = {}
        drawdown_preds = {}
        for p, feats in [('1h', std_1h), ('4h', std_4h), ('d1', std_d1)]:
            models_p = self.models.get(p, {})
            if 'cls' in models_p and 'up' not in models_p:
                ai_scores[p] = self.get_ai_score_cls(feats, models_p['cls'])
            else:
                cal_up = self.calibrators.get(p, {}).get('up')
                cal_down = self.calibrators.get(p, {}).get('down')
                if cal_up is None and cal_down is None:
                    ai_scores[p] = self.get_ai_score(
                        feats,
                        models_p['up'],
                        models_p['down'],
                    )
                else:
                    ai_scores[p] = self.get_ai_score(
                        feats,
                        models_p['up'],
                        models_p['down'],
                        cal_up,
                        cal_down,
                    )
            if 'vol' in self.models[p]:
                vol_preds[p] = self.get_vol_prediction(feats, self.models[p]['vol'])
            else:
                vol_preds[p] = None
            if 'rise' in self.models[p]:
                rise_preds[p] = self.get_reg_prediction(feats, self.models[p]['rise'])
            else:
                rise_preds[p] = None
            if 'drawdown' in self.models[p]:
                drawdown_preds[p] = self.get_reg_prediction(feats, self.models[p]['drawdown'])
            else:
                drawdown_preds[p] = None

        # d1 空头阈值特殊规则
        if ai_scores['d1'] < 0 and abs(ai_scores['d1']) < self.th_down_d1:
            ai_scores['d1'] = 0.0

        # ===== 2. 计算多因子部分的分数 =====
        # 若提供了未标准化的原始特征，则优先用于多因子逻辑计算，
        # 避免标准化偏移导致阈值判断失真
        fs = {
            '1h': self.get_factor_scores(raw_f1h, '1h'),
            '4h': self.get_factor_scores(raw_f4h, '4h'),
            'd1': self.get_factor_scores(raw_fd1, 'd1'),
        }

        # ===== 3. 使用当前因子权重 =====
        with self._lock:
            weights = self.current_weights.copy()

        # ===== 4. 单周期总分 =====
        scores = self.calc_factor_scores(ai_scores, fs, weights)

        # ===== 5. 本地因子修正 =====
        scores, local_details = self.apply_local_adjustments(
            scores,
            {'1h': raw_f1h, '4h': raw_f4h, 'd1': raw_fd1},
            fs,
            deltas,
            rise_preds.get('1h'),
            drawdown_preds.get('1h'),
        )

        ic_periods = {
            "1h": self.ic_scores.get("1h", 1.0),
            "4h": self.ic_scores.get("4h", 1.0),
            "d1": self.ic_scores.get("d1", 1.0),
        }
        w1, w4, w_d1 = self.get_ic_period_weights(ic_periods)

        # ===== 6. 多周期共振融合 =====
        fused_score, consensus_all, consensus_14, consensus_4d1 = self.fuse_multi_cycle(
            scores,
            (w1, w4, w_d1),
            local_details.get('strong_confirm_4h', False),
        )


        logic_score = fused_score
        env_score = 1.0
        risk_score = 1.0

        # 根据外部指标微调 fused_score
        if global_metrics is not None:
            dom = global_metrics.get('btc_dom_chg')
            if 'btc_dominance' in global_metrics:
                self.btc_dom_history.append(global_metrics['btc_dominance'])
                if len(self.btc_dom_history) >= 5:
                    short = np.mean(list(self.btc_dom_history)[-5:])
                    long = np.mean(self.btc_dom_history)
                    dom_diff = (short - long) / long if long else 0
                    if dom is None:
                        dom = dom_diff
                    else:
                        dom += dom_diff
            if dom is not None:
                if symbol and str(symbol).upper().startswith('BTC'):
                    fused_score *= 1 + 0.1 * dom
                else:
                    fused_score *= 1 - 0.1 * dom

            eth_dom = global_metrics.get('eth_dom_chg')
            if 'eth_dominance' in global_metrics:
                if not hasattr(self, 'eth_dom_history'):
                    self.eth_dom_history = deque(maxlen=500)
                self.eth_dom_history.append(global_metrics['eth_dominance'])
                if len(self.eth_dom_history) >= 5:
                    short_e = np.mean(list(self.eth_dom_history)[-5:])
                    long_e = np.mean(self.eth_dom_history)
                    dom_diff_e = (short_e - long_e) / long_e if long_e else 0
                    if eth_dom is None:
                        eth_dom = dom_diff_e
                    else:
                        eth_dom += dom_diff_e
            if eth_dom is not None:
                if symbol and str(symbol).upper().startswith('ETH'):
                    fused_score *= 1 + 0.1 * eth_dom
                else:
                    fused_score *= 1 + 0.05 * eth_dom
            btc_mcap = global_metrics.get('btc_mcap_growth')
            alt_mcap = global_metrics.get('alt_mcap_growth')
            mcap_g = global_metrics.get('mcap_growth')
            if symbol and str(symbol).upper().startswith('BTC'):
                base_mcap = btc_mcap if btc_mcap is not None else mcap_g
            else:
                base_mcap = alt_mcap if alt_mcap is not None else mcap_g
            if base_mcap is not None:
                fused_score *= 1 + 0.1 * base_mcap
            vol_c = global_metrics.get('vol_chg')
            if vol_c is not None:
                fused_score *= 1 + 0.05 * vol_c
            hot = global_metrics.get('hot_sector_strength')
            if hot is not None:

                corr = global_metrics.get('sector_corr')
                if corr is None:
                    hot_name = global_metrics.get('hot_sector')
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
        if open_interest is not None:
            oi_chg = open_interest.get('oi_chg')
            if oi_chg is not None:
                with self._lock:
                    cache["oi_change_history"].append(oi_chg)
                th_oi = self.get_dynamic_oi_threshold(pred_vol=vol_preds.get('1h'))
                fused_score, oi_overheat = self.apply_oi_overheat_protection(
                    fused_score, oi_chg, th_oi
                )

        # ===== 新指标：短周期动量与盘口失衡 =====
        raw1h = raw_f1h
        mom5 = raw1h.get('mom_5m_roll1h')
        mom15 = raw1h.get('mom_15m_roll1h')

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

        # ===== 7. 如果 fused_score 为 NaN，直接返回无信号 =====
        if fused_score is None or (isinstance(fused_score, float) and np.isnan(fused_score)):
            logging.debug("Fused score NaN, returning 0 signal")
            self._last_score = fused_score
            with self._lock:
                self._prev_raw["1h"] = raw_f1h
                self._prev_raw["4h"] = raw_f4h
                self._prev_raw["d1"] = raw_fd1
                for p, raw in [("1h", raw_f1h), ("4h", raw_f4h), ("d1", raw_fd1)]:
                    maxlen = 4 if p == "1h" else 2
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
                    'note': 'fused_score was NaN'
                }
            }

# ===== 7b. 计算动态阈值 =====
        atr_1h = raw_f1h.get('atr_pct_1h', features_1h.get('atr_pct_1h', 0))
        adx_1h = raw_f1h.get('adx_1h', features_1h.get('adx_1h', 0))
        funding_1h = raw_f1h.get('funding_rate_1h', features_1h.get('funding_rate_1h', 0)) or 0

        atr_4h = raw_f4h.get('atr_pct_4h', features_4h.get('atr_pct_4h', 0)) if raw_f4h else None
        adx_4h = raw_f4h.get('adx_4h', features_4h.get('adx_4h', 0)) if raw_f4h else None
        atr_d1 = raw_fd1.get('atr_pct_d1', features_d1.get('atr_pct_d1', 0)) if raw_fd1 else None
        adx_d1 = raw_fd1.get('adx_d1', features_d1.get('adx_d1', 0)) if raw_fd1 else None

        vix_p = None
        if global_metrics is not None:
            vix_p = global_metrics.get('vix_proxy')
        if vix_p is None and open_interest is not None:
            vix_p = open_interest.get('vix_proxy')

        regime = self.detect_market_regime(adx_1h, adx_4h or 0, adx_d1 or 0)
        cfg_th = self.signal_threshold_cfg
        base_th, rev_boost = self.dynamic_threshold(
            atr_1h,
            adx_1h,
            funding_1h,
            atr_4h=atr_4h,
            adx_4h=adx_4h,
            atr_d1=atr_d1,
            adx_d1=adx_d1,
            pred_vol=vol_preds.get('1h'),
            pred_vol_4h=vol_preds.get('4h'),
            pred_vol_d1=vol_preds.get('d1'),
            vix_proxy=vix_p,
            regime=regime,
            base=cfg_th.get('base_th', 0.12),
            reversal=bool(rev_dir),
            history_scores=cache["history_scores"],
        )
        base_th = max(base_th, 0.35)
        if rev_dir != 0:
            fused_score += rev_boost * rev_dir
            self._cooldown = 0
        # ===== 8. 资金费率惩罚 =====
        funding_conflicts = 0
        for p, raw_f in [('1h', raw_f1h), ('4h', raw_f4h), ('d1', raw_fd1)]:
            if raw_f is None:
                continue
            f_rate = raw_f.get(f'funding_rate_{p}', 0)
            if abs(f_rate) > 0.0005 and np.sign(f_rate) * np.sign(fused_score) < 0:
                penalty = min(abs(f_rate) * 20, 0.20)
                fused_score *= 1 - penalty
                funding_conflicts += 1
        if funding_conflicts >= 2:
            fused_score *= 0.85 ** funding_conflicts

        # ===== 9. 极端行情保护 =====
        crowding_factor = 1.0
        if not oi_overheat and all_scores_list is not None:
            crowding_factor = self.crowding_protection(all_scores_list, fused_score, base_th)
            fused_score *= crowding_factor
        if th_oi is not None:
            oi_crowd = float(np.clip((th_oi - 0.5) * 2, 0, 1))
            if oi_crowd > 0:
                mult = 1 - oi_crowd * 0.5
                logging.debug(
                    "oi threshold %.3f crowding factor %.3f for %s -> score *= %.3f",
                    th_oi,
                    oi_crowd,
                    coin,
                    mult,
                )
                fused_score *= mult
                crowding_factor *= mult
        risk_score = fused_to_risk(
            fused_score,
            logic_score,
            env_score,
            cap=self.risk_score_cap,
        )
        # 所有放大系数后写入历史
        with self._lock:
            cache["history_scores"].append(fused_score)

        # 细节信息留待方法末统一构建




        # ---- 新增：方向确认与多因子投票 ----
        vol_ratio_1h_4h = raw_f1h.get('vol_ratio_1h_4h')
        if vol_ratio_1h_4h is None and raw_f4h is not None:
            vol_ratio_1h_4h = raw_f4h.get('vol_ratio_1h_4h')
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
        vol_breakout_val = raw_f1h.get('vol_breakout_1h')
        vol_breakout_dir = 1 if vol_breakout_val and vol_breakout_val > 0 else 0

        th = self.ai_dir_eps
        if ai_scores['1h'] >= th:
            ai_dir = 1
        elif ai_scores['1h'] <= -th:
            ai_dir = -1
        else:
            ai_dir = 0

        vw = self.vote_weights
        vote = (
            vw.get('ob', 4) * ob_dir
            + vw.get('short_mom', 2) * short_mom_dir
            + vw.get('ai', self.vote_params['weight_ai']) * ai_dir
            + vw.get('vol_breakout', 1) * vol_breakout_dir
        )
        strong_confirm_vote = abs(vote) >= self.vote_params['strong_min']

        # ====== 票数置信度衰减 ======
        strong_min = self.vote_params['strong_min']
        conf_vote = sigmoid_confidence(vote, strong_min, 1)
        if abs(vote) >= strong_min:
            fused_score *= max(1, conf_vote)

        cfg_th_sig = self.signal_threshold_cfg
        grad_dir = sigmoid_dir(
            fused_score,
            base_th,
            cfg_th_sig.get('gamma', 0.05),
        )
        direction = 0 if grad_dir == 0 else int(np.sign(grad_dir))

        if self._cooldown > 0:
            self._cooldown -= 1

        if self._last_signal != 0 and direction != 0 and direction != self._last_signal:
            flip_th = max(base_th, self.flip_coeff * abs(self._last_score))
            if abs(fused_score) < flip_th or self._cooldown > 0:
                direction = self._last_signal
            else:
                self._cooldown = 2

        # 阶梯退出逻辑
        prev_vote = getattr(self, '_prev_vote', 0)

        # 多周期趋势与动量方向一致性过滤：至少两周期同向
        if direction != 0:
            align_count = 0
            for p in ('1h', '4h', 'd1'):
                if (
                    np.sign(fs[p]['trend']) == direction
                    and np.sign(fs[p]['momentum']) == direction
                ):
                    align_count += 1
            if align_count < 2:
                direction = 0

        # 区间突破检查暂时停用
        # if direction != 0 and regime == 'range':
        #     ch_pos = raw_f1h.get('channel_pos_1h', features_1h.get('channel_pos_1h'))
        #     if (direction == 1 and (ch_pos is None or ch_pos <= 1)) or (
        #         direction == -1 and (ch_pos is None or ch_pos >= 0)
        #     ):
        #         direction = 0


        # ===== 12. 仓位大小统一计算 =====
        base_coeff = (
            self.pos_coeff_range if regime == "range" else self.pos_coeff_trend
        )
        confidence_factor = 1.0
        if consensus_all:
            confidence_factor += 0.1
        if strong_confirm_vote:
            confidence_factor += 0.05
        vol_ratio = raw_f1h.get(
            'vol_ma_ratio_1h', features_1h.get('vol_ma_ratio_1h')
        )
        tier = 0.1 + base_coeff * abs(grad_dir)
        fused_score = float(np.clip(fused_score, -1, 1))
        pos_size, direction = self.compute_position_size(
            grad_dir=grad_dir,
            base_coeff=base_coeff,
            confidence_factor=confidence_factor,
            vol_ratio=vol_ratio,
            fused_score=fused_score,
            base_th=base_th,
            regime=regime,
            oi_overheat=oi_overheat,
            vol_p=vol_preds.get('1h'),
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
        )

        if oi_overheat:
            pos_size *= 0.5
            direction = 0

        # ===== 13. 止盈止损计算：使用 ATR 动态设置 =====
        price = features_1h.get('close', 0)
        if raw_features_4h is not None and 'atr_pct_4h' in raw_features_4h:
            atr_pct_4h = raw_features_4h['atr_pct_4h']
        else:
            atr_pct_4h = features_4h.get('atr_pct_4h', 0)
        atr_abs = np.hypot(atr_1h, atr_pct_4h) * price
        atr_abs = max(atr_abs, 0.005 * price)
        take_profit = stop_loss = None
        if direction != 0:
            take_profit, stop_loss = self.compute_tp_sl(
                price,
                atr_abs,
                direction,
                rise_pred=rise_preds.get('1h'),
                drawdown_pred=drawdown_preds.get('1h'),
            )

        # ===== 14. 最终返回 =====
        with self._lock:
            self._last_signal = int(np.sign(direction)) if direction else 0
            self._last_score = fused_score
            self._prev_vote = vote
            cache["_prev_raw"]["1h"] = raw_f1h
            cache["_prev_raw"]["4h"] = raw_f4h
            cache["_prev_raw"]["d1"] = raw_fd1
            for p, raw in [("1h", raw_f1h), ("4h", raw_f4h), ("d1", raw_fd1)]:
                maxlen = 4 if p == "1h" else 2
                cache["_raw_history"].setdefault(p, deque(maxlen=maxlen)).append(raw)

        filters = getattr(self, 'signal_filters', {"min_vote": 5, "confidence_vote": 0.2})
        confidence_vote = filters.get('confidence_vote', 0.2)
        if sigmoid_confidence(vote, self.vote_params['strong_min'], 1) < confidence_vote:
            direction, pos_size = 0, 0.0
            take_profit = stop_loss = None
        min_vote = filters.get('min_vote', 5)
        if abs(vote) < min_vote:
            direction, pos_size = 0, 0.0
            take_profit = stop_loss = None

        final_details = {
            'ai': {'1h': ai_scores['1h'], '4h': ai_scores['4h'], 'd1': ai_scores['d1']},
            'factors': {'1h': fs['1h'], '4h': fs['4h'], 'd1': fs['d1']},
            'scores': {'1h': scores['1h'], '4h': scores['4h'], 'd1': scores['d1']},
            'vote': {'value': vote, 'confidence': conf_vote, 'ob_th': ob_th},
            'protect': {
                'oi_overheat': oi_overheat,
                'oi_threshold': th_oi,
                'crowding_factor': crowding_factor,
                'funding_conflicts': funding_conflicts,
            },
            'env': {
                'logic_score': logic_score,
                'env_score': env_score,
                'risk_score': risk_score,
            },
            'exit': {
                'regime': regime,
                'reversal_flag': rev_dir,
                'dynamic_th_final': base_th,
            },
            'grad_dir': float(grad_dir),
            'pos_size': float(pos_size),
            'short_momentum': short_mom,
            'ob_imbalance': ob_imb,
            'vol_ratio': vol_ratio,
            'position_tier': tier,
            'confidence_factor': confidence_factor,
            'consensus_all': consensus_all,
            'consensus_14': consensus_14,
            'consensus_4d1': consensus_4d1,
        }
        final_details.update(local_details)

        logger.info(
            "[%s] base_th=%.3f, funding_conflicts=%d, fused=%.4f",
            symbol,
            base_th,
            funding_conflicts,
            fused_score,
        )
        return {
            'signal': direction,
            'score': fused_score,
            'position_size': pos_size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'details': final_details,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

import joblib
import numpy as np
import pandas as pd
from collections import Counter, deque
import threading
import logging
pd.set_option('future.no_silent_downcasting', True)

# 当订单簿动量与信号方向相反且超过该阈值时取消信号
ORDER_BOOK_MOM_THRESHOLD = 0.02


def softmax(x):
    """简单 softmax 实现"""
    arr = np.array(x, dtype=float)
    ex = np.exp(arr - np.nanmax(arr))
    return ex / ex.sum()

class RobustSignalGenerator:
    """多周期 AI + 多因子 融合信号生成器。

    - 支持动态阈值与极端行情防护
    - 可通过 :func:`update_ic_scores` 读取历史数据并计算因子 IC
      用于动态调整权重
    """

    def __init__(self, model_paths, *, feature_cols_1h, feature_cols_4h, feature_cols_d1, history_window=3000, symbol_categories=None):
        # 加载AI模型，同时保留训练时的 features 列名
        self.models = {}
        for period, path_dict in model_paths.items():
            self.models[period] = {}
            for direction, path in path_dict.items():
                loaded = joblib.load(path)
                # loaded = {"model": LGBMClassifier, "features": [...训练时的列名列表...]}
                # >>>>> 修改点：同时保存 model 和 features
                self.models[period][direction] = {
                    "model":   loaded["model"],
                    "features": loaded["features"]
                }

        # 保存各时间周期对应的特征列（但后面不再直接用它；实际推理要以 loaded["features"] 为准）
        self.feature_cols_1h = feature_cols_1h
        self.feature_cols_4h = feature_cols_4h
        self.feature_cols_d1 = feature_cols_d1

        # 静态因子权重（后续可由动态IC接口进行更新）
        _base_weights = {
            'ai': 0.15,
            'trend': 0.2,
            'momentum': 0.2,
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

        # 初始化各因子对应的IC分数（均设为1，后续可动态更新）
        self.ic_scores = {k: 1 for k in self.base_weights.keys()}
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


        # 当多个信号方向过于集中时，用于滤除极端行情（最大同向信号比例阈值）
        self.max_same_direction_rate = 0.85

        # 保存上一次生成的信号，供滞后阈值逻辑使用
        self._last_signal = 0

        # 缓存上一周期的原始特征，便于计算指标变化量
        self._prev_raw = {p: None for p in ("1h", "4h", "d1")}

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
        if name == "_prev_raw":
            val = {p: None for p in ("1h", "4h", "d1")}
            setattr(self, name, val)
            return val
        raise AttributeError(name)


    def get_dynamic_oi_threshold(self, pred_vol=None, base=0.5, quantile=0.9):
        """根据历史 OI 变化率及预测波动率自适应阈值"""
        th = base
        if len(self.oi_change_history) > 30:
            th = float(np.quantile(np.abs(self.oi_change_history), quantile))
        if pred_vol is not None:
            th += min(0.1, abs(pred_vol) * 0.5)
        return th

    def detect_market_regime(self, adx1, adx4, adxd):
        """简易市场状态判别：根据平均ADX判断震荡或趋势"""
        avg_adx = np.nanmean([adx1, adx4, adxd])
        return "trend" if avg_adx >= 25 else "range"

    def calc_period_weights(self, adx1, adx4, adxd):
        """根据各周期ADX分配权重"""
        w1 = 0.6 + 0.4 * min(adx1, 50) / 50
        w4 = 0.3 + 0.4 * min(adx4, 50) / 50
        wd = 0.1 + 0.4 * min(adxd, 50) / 50
        total = w1 + w4 + wd
        return w1 / total, w4 / total, wd / total

    def set_symbol_categories(self, mapping):
        """更新币种与板块的映射"""
        self.symbol_categories = {k.upper(): v for k, v in mapping.items()}


    def compute_tp_sl(self, price, atr, direction, tp_mult=1.5, sl_mult=1.0):
        """计算止盈止损价格"""
        if price is None or price <= 0:
            return None, None            # 价格异常直接放弃
        if atr is None or atr == 0:
            atr = 0.005 * price

        # 限制倍数范围，防止 ATR 极端波动导致止盈/止损过远或过近
        tp_mult = float(np.clip(tp_mult, 0.5, 3.0))
        sl_mult = float(np.clip(sl_mult, 0.5, 2.0))

        if direction == 1:
            take_profit = price + tp_mult * atr
            stop_loss = price - sl_mult * atr
        else:
            take_profit = price - tp_mult * atr
            stop_loss = price + sl_mult * atr
        return float(take_profit), float(stop_loss)

    @staticmethod
    def _calc_deltas(curr: dict, prev: dict, keys: list, scale: float = 1.0) -> dict:
        """返回 {f"{k}_delta": (curr[k] - prev[k]) * scale}；若 prev 为 None 则全 0."""
        deltas = {}
        if prev is None:
            for k in keys:
                deltas[f"{k}_delta"] = 0.0
            return deltas

        for k in keys:
            c_val = curr.get(k, 0)
            p_val = prev.get(k, 0)
            if c_val is None or p_val is None:
                deltas[f"{k}_delta"] = 0.0
            else:
                deltas[f"{k}_delta"] = (c_val - p_val) * scale
        return deltas

    @staticmethod
    def _apply_delta_boost(score: float, deltas: dict) -> float:
        """根据指标变化量对得分进行微调"""
        mom_boost = 0.0
        if deltas.get("rsi_1h_delta", 0) > 5 and np.sign(score) == 1:
            mom_boost += 0.05
        if deltas.get("rsi_1h_delta", 0) < -5 and np.sign(score) == -1:
            mom_boost += 0.05
        if deltas.get("macd_hist_1h_delta", 0) * score > 0:
            mom_boost += 0.05
        return score * (1 + mom_boost)

    def ma_cross_logic(self, features: dict, sma_20_1h_prev=None) -> int:
        """根据1h MA5 与 MA20 判断多空方向"""

        sma5 = features.get('sma_5_1h')
        sma20 = features.get('sma_20_1h')
        ratio = features.get('ma_ratio_5_20')
        if sma5 is None or sma20 is None or ratio is None:
            return 0

        slope = (
            (sma20 - sma_20_1h_prev) / sma_20_1h_prev
            if sma_20_1h_prev
            else 0.0
        )

        if ratio > 1.02 and slope > 0:
            return 1
        if ratio < 0.98 and slope < 0:
            return -1
        return 0

    def get_position_size(self, score, base=0.1, coeff=0.9):
        """根据当前得分及策略回撤计算仓位大小"""
        drawdown = getattr(self, "_equity_drawdown", 0.0)
        if not 0 <= drawdown <= 1:
            drawdown = 0.0
        pos = base + coeff * min(abs(score), 1.0) * (1 - drawdown)
        return float(np.clip(pos, 0.0, 1.0))

    # >>>>> 修改：改写 get_ai_score，让它自动从 self.models[...]["features"] 中取“训练时列名”
    def get_ai_score(self, features, model_up, model_down):
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
                logging.warning("get_ai_score missing columns: %s", missing)
            df = pd.DataFrame(row)
            return df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)

        X_up = _build_df(model_up)
        X_down = _build_df(model_down)
        prob_up = model_up["model"].predict_proba(X_up)[:, 1]
        prob_down = model_down["model"].predict_proba(X_down)[:, 1]
        denom = prob_up + prob_down
        ai_score = np.where(denom == 0, 0.0, (prob_up - prob_down) / denom)
        ai_score = np.clip(ai_score, -1.0, 1.0)
        if ai_score.size == 1:
            return float(ai_score[0])
        return ai_score

    def get_vol_prediction(self, features, model_dict):
        """根据回归模型预测未来波动率"""
        lgb_model = model_dict["model"]
        train_cols = model_dict["features"]

        row_data = {col: [features.get(col, 0)] for col in train_cols}
        X_df = pd.DataFrame(row_data)
        X_df = X_df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)
        return float(lgb_model.predict(X_df)[0])

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
        )

        momentum_raw = (
            (safe(f'rsi_{period}', 50) - 50) / 50
            + safe(f'willr_{period}', -50) / 50
            + np.tanh(safe(f'macd_hist_{period}', 0) * 5)
            + np.tanh(safe(f'rsi_slope_{period}', 0) * 10)
            + (safe(f'mfi_{period}', 50) - 50) / 50
        )

        volatility_raw = (
            np.tanh(safe(f'atr_pct_{period}', 0) * 8)
            + np.tanh(safe(f'bb_width_{period}', 0) * 2)
            + np.tanh(safe(f'donchian_delta_{period}', 0) * 5)
            + np.tanh(safe(f'hv_7d_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'hv_14d_{period}', 0) * 5)
            + 0.5 * np.tanh(safe(f'hv_30d_{period}', 0) * 5)
        )

        volume_raw = (
            np.tanh(safe(f'vol_ma_ratio_{period}', 0))
            + np.tanh(safe(f'obv_delta_{period}', 0) / 1e5)
            + np.tanh(safe(f'vol_roc_{period}', 0) / 5)
            + np.tanh(safe(f'rsi_mul_vol_ma_ratio_{period}', 0) / 100)
            + np.tanh((safe(f'buy_sell_ratio_{period}', 1) - 1) * 2)
            + np.tanh(safe(f'vol_profile_density_{period}', 0) / 10)
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
        if abs(f_rate) > 0.05:
            funding_raw = -np.tanh(f_rate * 100)
        else:
            funding_raw = np.tanh(f_rate * 100)
        funding_raw += np.tanh(f_anom * 50)

        return {
            'trend': np.tanh(trend_raw),
            'momentum': np.tanh(momentum_raw),
            'volatility': np.tanh(volatility_raw),
            'volume': np.tanh(volume_raw),
            'sentiment': np.tanh(sentiment_raw),
            'funding': np.tanh(funding_raw),
        }

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

    def dynamic_weight_update(self, halflife=50):
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
                    weights = np.exp(decay * np.arange(len(arr)))
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
                raw[k] = max(0.0, w)

            total = sum(raw.values()) or 1.0
            self.current_weights = {k: v / total for k, v in raw.items()}
            return self.current_weights

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
        base=0.10,
        min_thres=0.06,
        max_thres=0.25,
        regime=None,
    ):
        """根据历史 ATR、ADX、预测波动率及恐慌指数动态计算阈值"""

        thres = base

        # ===== 波动性贡献 =====
        thres += min(0.08, abs(atr) * 3)
        if atr_4h is not None:
            thres += 0.5 * min(0.08, abs(atr_4h) * 3)
        if atr_d1 is not None:
            thres += 0.25 * min(0.08, abs(atr_d1) * 3)
        if pred_vol is not None:
            thres += min(0.05, abs(pred_vol) * 3)
        if pred_vol_4h is not None:
            thres += 0.5 * min(0.05, abs(pred_vol_4h) * 3)
        if pred_vol_d1 is not None:
            thres += 0.25 * min(0.05, abs(pred_vol_d1) * 3)

        # ===== 趋势强度贡献 =====
        thres += min(0.08, max(adx - 20, 0) * 0.004)
        if adx_4h is not None:
            thres += 0.5 * min(0.08, max(adx_4h - 20, 0) * 0.004)
        if adx_d1 is not None:
            thres += 0.25 * min(0.08, max(adx_d1 - 20, 0) * 0.004)

        # ===== 资金费率贡献 =====
        thres += min(0.05, abs(funding) * 5)

        # ===== 恐慌指数贡献 =====
        if vix_proxy is not None:
            thres += min(0.05, max(0.0, vix_proxy) * 0.05)

        # ===== 历史分位阈值补充 =====
        if len(self.history_scores) > 100:
            quantile_th = np.quantile(np.abs(self.history_scores), 0.92)
            thres = max(thres, quantile_th)

        if regime == "trend":
            thres += 0.02
        elif regime == "range":
            thres -= 0.02

        return max(min_thres, min(max_thres, thres))

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

        return float(np.clip(fused_score, -1, 1))

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

    def crowding_protection(self, scores, current_score):
        """根据同向排名抑制过度拥挤的信号，返回衰减系数"""
        if not scores:
            return 1.0

        arr = np.array(list(scores) + [current_score], dtype=float)
        signs = [s for s in np.sign(arr[:-1]) if s != 0]
        total = len(signs)
        if total == 0:
            return 1.0
        pos_counts = Counter(signs)
        dominant_dir, cnt = pos_counts.most_common(1)[0]

        factor = 1.0
        if total > 0 and cnt / total > self.max_same_direction_rate and np.sign(current_score) == dominant_dir:
            factor = 0.5

        ranks = pd.Series(np.abs(arr)).rank(pct=True).iloc[-1]
        if ranks >= 0.9 and np.sign(current_score) == dominant_dir:
            factor = min(factor, 0.7)

        return factor

    def apply_oi_overheat_protection(self, fused_score, oi_chg, th_oi):
        """根据 OI 变化率奖励或惩罚分数"""
        oi_overheat = False
        if th_oi is None:
            return fused_score, oi_overheat
        if abs(oi_chg) < th_oi:
            fused_score *= 1 + 0.08 * oi_chg
        else:
            logging.debug("OI overheat detected: %.4f", oi_chg)
            fused_score *= 0.8
            oi_overheat = True
        return fused_score, oi_overheat

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
            - features_d1: dict，当前 1d 周期下的全部特征键值对（已标准化）
            - all_scores_list: list，可选，当前所有币种的 fused_score 列表，用于极端行情保护
            - raw_features_1h: dict，可选，未标准化的 1h 特征
            - raw_features_4h: dict，可选，未标准化的 4h 特征；其中 atr_pct_4h 为实际
              比例（如 0.05 表示 5%），在计算止盈止损和指标计算时会优先使用
            - raw_features_d1: dict，可选，未标准化的 1d 特征
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

        raw_f1h = raw_features_1h or features_1h
        raw_f4h = raw_features_4h or features_4h
        raw_fd1 = raw_features_d1 or features_d1

        CORE_KEYS = [
            "rsi_1h", "macd_hist_1h", "ema_diff_1h",
            "atr_pct_1h", "vol_ma_ratio_1h", "funding_rate_1h",
        ]
        deltas_1h = self._calc_deltas(raw_f1h, self._prev_raw["1h"], CORE_KEYS)

        # ===== 1. 计算 AI 部分的分数（映射到 [-1, 1]） =====
        ai_scores = {}
        vol_preds = {}
        for p, feats in [('1h', features_1h), ('4h', features_4h), ('d1', features_d1)]:
            ai_scores[p] = self.get_ai_score(feats, self.models[p]['up'], self.models[p]['down'])
            if 'vol' in self.models[p]:
                vol_preds[p] = self.get_vol_prediction(feats, self.models[p]['vol'])
            else:
                vol_preds[p] = None

        # ===== 2. 计算多因子部分的分数 =====
        # 若提供了未标准化的原始特征，则优先用于多因子逻辑计算，
        # 避免标准化偏移导致阈值判断失真
        fs = {
            '1h': self.get_factor_scores(raw_features_1h or features_1h, '1h'),
            '4h': self.get_factor_scores(raw_features_4h or features_4h, '4h'),
            'd1': self.get_factor_scores(raw_features_d1 or features_d1, 'd1')
        }

        # ===== 3. 动态权重更新 =====
        weights = self.dynamic_weight_update()

        # ===== 4. 合并 AI 与多因子分数，得到各周期总分 =====
        score_1h = self.combine_score(ai_scores['1h'], fs['1h'], weights)
        score_4h = self.combine_score(ai_scores['4h'], fs['4h'], weights)
        score_d1 = self.combine_score(ai_scores['d1'], fs['d1'], weights)

        # 根据关键指标的变化量微调 1h 得分
        score_1h = self._apply_delta_boost(score_1h, deltas_1h)

        # ===== 5. 判断 4h 强确认条件 =====
        strong_confirm = (
            (fs['4h']['trend'] > 0 and fs['4h']['momentum'] > 0 and fs['4h']['volatility'] > 0 and score_4h > 0) or
            (fs['4h']['trend'] < 0 and fs['4h']['momentum'] < 0 and fs['4h']['volatility'] < 0 and score_4h < 0)
        )

        # ===== 6. 多周期共振：使用 consensus_check =====
        consensus_dir = self.consensus_check(score_1h, score_4h, score_d1)
        consensus_all = (
            consensus_dir != 0 and np.sign(score_1h) == np.sign(score_4h) == np.sign(score_d1)
        )
        consensus_14 = (
            consensus_dir != 0 and np.sign(score_1h) == np.sign(score_4h) and not consensus_all
        )

        adx1 = (raw_features_1h or features_1h).get('adx_1h', 0)
        adx4 = (raw_features_4h or features_4h).get('adx_4h', 0)
        adxd = (raw_features_d1 or features_d1).get('adx_d1', 0)
        w1, w4, w_d1 = self.calc_period_weights(adx1, adx4, adxd)

        # ---- 额外逻辑：情绪与交易量等修正 ----
        scores = {'1h': score_1h, '4h': score_4h, 'd1': score_d1}
        for p in scores:
            sent = fs[p]['sentiment']
            if sent < -0.5:
                old = scores[p]
                scores[p] *= 1.5
                logging.debug(
                    "sentiment %.2f < -0.5 on %s -> score_%s %.3f -> %.3f",
                    sent, p, p, old, scores[p]
                )

        sentiment_combined = (
            w1 * fs['1h']['sentiment']
            + w4 * fs['4h']['sentiment']
            + w_d1 * fs['d1']['sentiment']
        )
        if sentiment_combined <= -0.25:
            logging.debug(
                "combined sentiment %.3f <= -0.25 -> cap positive scores",
                sentiment_combined,
            )
            for p in scores:
                if scores[p] > 0:
                    scores[p] = 0.0

        # volume guard-rail
        raw1h = raw_features_1h or features_1h
        raw4h = raw_features_4h or features_4h
        vol_ratio_1h = raw1h.get('vol_ma_ratio_1h')
        vol_roc_1h = raw1h.get('vol_roc_1h')
        if (vol_ratio_1h is not None and vol_ratio_1h < 0.8) or (
            vol_roc_1h is not None and vol_roc_1h < -20
        ):
            old = scores['1h']
            scores['1h'] -= 0.15
            logging.debug(
                "volume guard 1h ratio=%.3f roc=%.3f -> %.3f",
                vol_ratio_1h, vol_roc_1h, scores['1h']
            )

        if raw4h is not None:
            vol_ratio_4h = raw4h.get('vol_ma_ratio_4h')
            vol_roc_4h = raw4h.get('vol_roc_4h')
            if (vol_ratio_4h is not None and vol_ratio_4h < 0.8) or (
                vol_roc_4h is not None and vol_roc_4h < -10
            ):
                old = scores['4h']
                scores['4h'] -= 0.15
                logging.debug(
                    "volume guard 4h ratio=%.3f roc=%.3f -> %.3f",
                    vol_ratio_4h, vol_roc_4h, scores['4h']
                )
        else:
            vol_roc_4h = None

        if vol_roc_1h is not None and vol_roc_1h > 100:
            scores["1h"] -= 0.10
        if vol_roc_4h is not None and vol_roc_4h > 50:
            scores["4h"] -= 0.10
        # 惩罚后再裁剪
        for p in ("1h", "4h"):
            scores[p] = float(np.clip(scores[p], -1, 1))

        for p, raw_f in [('1h', raw1h), ('4h', raw4h), ('d1', raw_features_d1 or features_d1)]:
            anom = raw_f.get(f'funding_rate_anom_{p}', 0)
            super_dir = raw_f.get(f'supertrend_dir_{p}', 0)
            bias = np.sign(anom)
            if bias != 0 and super_dir != 0 and bias != super_dir:
                if p == '1h':
                    pre = scores['1h']
                    scores['1h'] -= 0.1
                    logging.debug(
                        "funding bias opposes supertrend on 1h %.3f -> %.3f",
                        pre, scores['1h']

                    )
                elif p == '4h':
                    pre = scores['4h']
                    scores['4h'] -= 0.1
                    logging.debug(
                        "funding bias opposes supertrend on 4h %.3f -> %.3f",
                        pre, scores['4h']
                    )
                else:
                    pre = scores['d1']
                    scores['d1'] -= 0.1
                    logging.debug(
                        "funding bias opposes supertrend on d1 %.3f -> %.3f",
                        pre, scores['d1']
                    )

        macd_diff = raw1h.get('macd_hist_diff_1h_4h')
        rsi_diff = raw1h.get('rsi_diff_1h_4h')
        if (
            macd_diff is not None
            and rsi_diff is not None
            and macd_diff < 0
            and rsi_diff < -8
        ):
            if strong_confirm:
                logging.debug(
                    "momentum misalign macd_diff=%.3f rsi_diff=%.3f -> strong_confirm=False",
                    macd_diff,
                    rsi_diff,
                )
            strong_confirm = False

        # ensure scores remain within [-1, 1] after adjustments
        for p in scores:
            scores[p] = float(np.clip(scores[p], -1, 1))

        score_1h, score_4h, score_d1 = scores['1h'], scores['4h'], scores['d1']

        if consensus_all:
            fused_score = w1 * score_1h + w4 * score_4h + w_d1 * score_d1
            if strong_confirm:
                fused_score *= 1.15
        elif consensus_14:
            total = w1 + w4
            fused_score = (w1 / total) * score_1h + (w4 / total) * score_4h
            if strong_confirm:
                fused_score *= 1.10
        else:
            fused_score = score_1h

        fused_score = float(np.clip(fused_score, -1, 1))

        prev_ma20 = (raw_features_1h or features_1h).get('sma_20_1h_prev')
        ma_dir = self.ma_cross_logic(raw_features_1h or features_1h, prev_ma20)
        if ma_dir != 0:
            if np.sign(fused_score) == 0:
                fused_score += 0.1 * ma_dir
            elif np.sign(fused_score) == ma_dir:
                fused_score *= 1.1
            else:
                fused_score *= 0.7

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
        oi_overheat = False
        th_oi = None
        if open_interest is not None:
            oi_chg = open_interest.get('oi_chg')
            if oi_chg is not None:
                with self._lock:
                    self.oi_change_history.append(oi_chg)
                th_oi = self.get_dynamic_oi_threshold(pred_vol=vol_preds.get('1h'))
                fused_score, oi_overheat = self.apply_oi_overheat_protection(
                    fused_score, oi_chg, th_oi
                )

        # ===== 新指标：短周期动量与盘口失衡 =====
        raw1h = raw_features_1h or features_1h
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
            self._prev_raw["1h"] = raw_f1h
            self._prev_raw["4h"] = raw_f4h
            self._prev_raw["d1"] = raw_fd1
            return {
                'signal': 0,
                'score': float('nan'),
                'position_size': 0.0,
                'take_profit': None,
                'stop_loss': None,
                'details': {
                    'ai_1h': ai_scores['1h'],   'ai_4h': ai_scores['4h'],   'ai_d1': ai_scores['d1'],
                    'factors_1h': fs['1h'],     'factors_4h': fs['4h'],     'factors_d1': fs['d1'],
                    'score_1h': score_1h,       'score_4h': score_4h,       'score_d1': score_d1,
                    'strong_confirm': strong_confirm,
                    'consensus_14': consensus_14, 'consensus_all': consensus_all,
                    'vol_pred_1h': vol_preds.get('1h'),
                    'vol_pred_4h': vol_preds.get('4h'),
                    'vol_pred_d1': vol_preds.get('d1'),
                    'note': 'fused_score was NaN'
                }
            }

        # ===== 8. 极端行情保护 =====
        crowding_factor = 1.0
        if all_scores_list is not None:
            crowding_factor = self.crowding_protection(all_scores_list, fused_score)
            fused_score *= crowding_factor
        if th_oi is not None:
            oi_crowd = float(np.clip((th_oi - 0.5) * 2, 0, 1))
            if oi_crowd > 0:
                mult = 1 - oi_crowd * 0.5
                logging.debug(
                    "oi threshold %.3f crowding factor %.3f -> score *= %.3f",
                    th_oi,
                    oi_crowd,
                    mult,
                )
                fused_score *= mult
                crowding_factor *= mult

        # 所有放大系数后再最终裁剪
        fused_score = float(np.clip(fused_score, -1, 1))
        with self._lock:
            self.history_scores.append(fused_score)
        
        # ===== 9. 准备 details，用于回测与调试 =====
        details = {
            'ai_1h': ai_scores['1h'],   'ai_4h': ai_scores['4h'],   'ai_d1': ai_scores['d1'],
            'factors_1h': fs['1h'],     'factors_4h': fs['4h'],     'factors_d1': fs['d1'],
            'score_1h': score_1h,       'score_4h': score_4h,       'score_d1': score_d1,
            'strong_confirm': strong_confirm,
            'consensus_14': consensus_14, 'consensus_all': consensus_all,
            'vol_pred_1h': vol_preds.get('1h'),
            'vol_pred_4h': vol_preds.get('4h'),
            'vol_pred_d1': vol_preds.get('d1'),
            'oi_overheat': oi_overheat,
            'oi_threshold': th_oi,
            'crowding_factor': crowding_factor,
            'short_momentum': short_mom,
            'ob_imbalance': ob_imb,
            'ma_cross': ma_dir,
        }

        # ===== 11. 动态阈值过滤，调用已有 dynamic_threshold =====
        raw_f1h = raw_features_1h or features_1h
        raw_f4h = raw_features_4h or features_4h
        raw_fd1 = raw_features_d1 or features_d1

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
        th = self.dynamic_threshold(
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
        )

        base_th = th
        dyn_base = 0.25
        v_ratio = raw_f1h.get('vol_ma_ratio_1h')
        if v_ratio is not None and v_ratio < 0.8:
            dyn_base += 0.1
        if fs['1h']['sentiment'] < -0.5:
            dyn_base += 0.1
        if regime == "range" and consensus_all == 0:
            dyn_base += 0.05
        if base_th < dyn_base:
            logging.debug(
                "base threshold %.3f adjusted to %.3f due to vol/sentiment",
                base_th,
                dyn_base,
            )
            base_th = dyn_base
        details['regime'] = regime
        details['base_threshold'] = base_th

        if regime == "range" and consensus_dir == 0:
            logging.debug("Range regime without consensus -> no trade")
            self._last_signal = 0
            self._last_score = fused_score
            self._prev_raw["1h"] = raw_f1h
            self._prev_raw["4h"] = raw_f4h
            self._prev_raw["d1"] = raw_fd1
            return {
                'signal': 0,
                'score': fused_score,
                'position_size': 0.0,
                'take_profit': None,
                'stop_loss': None,
                'details': details,
            }

        if ob_imb is not None:
            details['order_book_imbalance'] = float(ob_imb)

        direction = 0
        if abs(fused_score) >= base_th:
            direction = int(np.sign(fused_score))

        if self._last_signal != 0 and direction != 0 and direction != self._last_signal:
            last_score = getattr(self, '_last_score', 0.0)
            flip_th = base_th + max(0.05, 0.5 * abs(last_score))
            if abs(fused_score) < max(flip_th, 1.2 * atr_1h):
                logging.debug("Flip prevented: last=%s current=%.3f", self._last_signal, fused_score)
                direction = self._last_signal

        if direction == 0:
            self._last_signal = 0
            self._last_score = fused_score
            self._prev_raw["1h"] = raw_f1h
            self._prev_raw["4h"] = raw_f4h
            self._prev_raw["d1"] = raw_fd1
            return {
                'signal': 0,
                'score': fused_score,
                'position_size': 0.0,
                'take_profit': None,
                'stop_loss': None,
                'details': details
            }

        if ob_imb is not None:
            details['order_book_imbalance'] = float(ob_imb)
            if abs(ob_imb) > 0.1 and np.sign(ob_imb) != np.sign(fused_score):
                logging.debug("Order book imbalance opposes score: %.4f", ob_imb)
                self._last_score = fused_score
                self._last_signal = 0
                self._prev_raw["1h"] = raw_f1h
                self._prev_raw["4h"] = raw_f4h
                self._prev_raw["d1"] = raw_fd1
                return {
                    'signal': 0,
                    'score': fused_score,
                    'position_size': 0.0,
                    'take_profit': None,
                    'stop_loss': None,
                    'details': details,
                }

        # ===== 12. 仓位大小按连续得分映射 =====
        coeff = 0.6 if consensus_all == 0 else 0.9
        pos_size = 0.1 + coeff * abs(fused_score)

        vol_ratio = raw_f1h.get('vol_ma_ratio_1h', features_1h.get('vol_ma_ratio_1h'))
        details['vol_ratio'] = vol_ratio
        if (
            regime == "range"
            and vol_ratio is not None
            and vol_ratio < 0.3
            and abs(fused_score) < base_th + 0.02
        ):
            direction = 0
            pos_size = 0.0

        if oi_overheat:
            pos_size *= 0.7

        vol_p = vol_preds.get('1h')
        if vol_p is not None:
            pos_size *= max(0.4, 1 - min(0.6, vol_p))


        details['order_book_momentum'] = ob_imb
        if (
            ob_imb is not None
            and direction != 0
            and abs(ob_imb) > ORDER_BOOK_MOM_THRESHOLD
            and np.sign(ob_imb) != direction
        ):
            logging.debug(
                "Direction canceled by order book imbalance %.4f", ob_imb
            )
            direction = 0
            pos_size = 0.0

        if direction == 1 and scores["4h"] < -0.7:
            direction, pos_size = 0, 0.0
        elif direction == -1 and scores["4h"] > 0.7:
            direction, pos_size = 0, 0.0

        # ===== 13. 止盈止损计算：使用 ATR 动态设置 =====
        price = features_1h.get('close', 0)
        if raw_features_4h is not None and 'atr_pct_4h' in raw_features_4h:
            atr_pct_4h = raw_features_4h['atr_pct_4h']
        else:
            atr_pct_4h = features_4h.get('atr_pct_4h', 0)
        atr_abs = max(atr_1h, atr_pct_4h) * price
        tp_dir = direction if direction != 0 else 1
        take_profit, stop_loss = self.compute_tp_sl(price, atr_abs, tp_dir)

        # ===== 14. 最终返回 =====
        self._last_signal = direction
        self._last_score = fused_score
        self._prev_raw["1h"] = raw_f1h
        self._prev_raw["4h"] = raw_f4h
        self._prev_raw["d1"] = raw_fd1
        return {
            'signal': direction,
            'score': fused_score,
            'position_size': pos_size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'details': details
        }

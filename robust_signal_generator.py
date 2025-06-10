import joblib
import numpy as np
import pandas as pd
from collections import Counter, deque
pd.set_option('future.no_silent_downcasting', True)

class RobustSignalGenerator:
    """多周期 AI + 多因子 融合信号生成器。

    - 支持动态阈值与极端行情防护
    - 可通过 :func:`update_ic_scores` 读取历史数据并计算因子 IC
      用于动态调整权重
    """

    def __init__(self, model_paths, *, feature_cols_1h, feature_cols_4h, feature_cols_d1, history_window=300):
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
        self.base_weights = {
            'ai': 0.15,
            'trend': 0.2,
            'momentum': 0.2,
            'volatility': 0.2,
            'volume': 0.1,
            'sentiment': 0.05,
            'funding': 0.05
        }

        # 当前权重，初始与 base_weights 相同
        self.current_weights = self.base_weights.copy()

        # 初始化各因子对应的IC分数（均设为1，后续可动态更新）
        self.ic_scores = {k: 1 for k in self.base_weights.keys()}

        # 用于存储历史融合得分，方便计算动态阈值（最大长度由 history_window 指定）
        self.history_scores = deque(maxlen=history_window)

        # 当多个信号方向过于集中时，用于滤除极端行情（最大同向信号比例阈值）
        self.max_same_direction_rate = 0.85

    @staticmethod
    def _score_from_proba(p):
        # 概率[0,1]映射到[-1,1]
        return 2 * p - 1

    def compute_tp_sl(self, price, atr, direction, tp_mult=1.5, sl_mult=1.0):
        """
        计算止盈止损价格
        :param price: 当前价格
        :param atr:   ATR绝对值（如4h的ATR*close）
        :param direction: 1=多头，-1=空头
        :param tp_mult: 止盈倍数
        :param sl_mult: 止损倍数
        :return: (take_profit, stop_loss)
        """
        if direction == 1:
            take_profit = price + tp_mult * atr
            stop_loss   = price - sl_mult * atr
        else:
            take_profit = price - tp_mult * atr
            stop_loss   = price + sl_mult * atr
        return take_profit, stop_loss

    # >>>>> 修改：改写 get_ai_score，让它自动从 self.models[...]["features"] 中取“训练时列名”
    def get_ai_score(self, features, model_dict):
        """
        features: 当前币种的特征字典（key: 列名，value: 数值）
        model_dict: 形如 {"model": LGBMClassifier, "features": [...训练时列...]}

        返回：AI 置信度映射到 [-1,1] 的综合得分
        """
        lgb_model = model_dict["model"]
        train_cols = model_dict["features"]

        row_data = {col: [features.get(col, 0)] for col in train_cols}
        X_df = pd.DataFrame(row_data)
        # === 这里加一行，强制所有特征为 float 类型，防止 dtype 报错 ===
        X_df = X_df.replace(['', None], np.nan).infer_objects(copy=False).astype(float)

        proba_up = lgb_model.predict_proba(X_df)[0][1]
        return self._score_from_proba(proba_up)

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

        def safe(key: str, default=0):
            """如果值缺失或为 NaN，返回 default。"""
            v = features.get(key, default)
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
        return self.ic_scores

    def dynamic_weight_update(self):
        # IC驱动动态权重分配（可扩展为每因子滑窗IC）
        ic_arr = np.array([max(0, v) for v in self.ic_scores.values()])
        base_arr = np.array([self.base_weights[k] for k in self.ic_scores.keys()])

        combined = ic_arr * base_arr
        if combined.sum() > 0:
            w = combined / combined.sum()
        else:
            w = base_arr / base_arr.sum()

        self.current_weights = dict(zip(self.ic_scores.keys(), w))
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
        base=0.10,
        min_thres=0.06,
        max_thres=0.25,
    ):
        """根据历史 ATR、ADX 以及预测波动率动态计算阈值"""

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

        # ===== 历史分位阈值补充 =====
        if len(self.history_scores) > 100:
            quantile_th = np.quantile(np.abs(self.history_scores), 0.92)
            thres = max(thres, quantile_th)

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

    def crowding_protection(self, signal_list):
        # 极端行情保护：若同方向信号占比过高，自动降权或减仓
        pos_counts = Counter(signal_list)
        total = sum(pos_counts.values())
        for direction in [-1, 1]:
            if total > 0 and pos_counts[direction] / total > self.max_same_direction_rate:
                return direction
        return 0

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
            - symbol: str，可选，当前币种，如 'BTCUSDT'
        输出：
            一个 dict，包含 'signal'、'score'、'position_size'、'take_profit'、'stop_loss' 和 'details'
        说明：若传入 raw_features_*，则多因子计算与动态阈值、止盈止损均使用原始数据，
              标准化后的 features_* 仅用于 AI 模型预测。
        """

        # ===== 1. 计算 AI 部分的分数（映射到 [-1, 1]） =====
        ai_scores = {}
        vol_preds = {}
        for p, feats in [('1h', features_1h), ('4h', features_4h), ('d1', features_d1)]:
            score_up = self.get_ai_score(feats, self.models[p]['up'])
            score_down = self.get_ai_score(feats, self.models[p]['down'])
            ai_scores[p] = 0.5 * (score_up - score_down)
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
        trend_strength = np.mean([adx1, adx4, adxd])
        t = max(0.0, min(50.0, trend_strength)) / 50.0
        w4 = 0.2 + 0.1 * t
        w_d1 = 0.1 + 0.1 * t
        w1 = 1 - w4 - w_d1

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

        # 根据外部指标微调 fused_score
        if global_metrics is not None:
            dom = global_metrics.get('btc_dom_chg')
            if dom is not None:
                if symbol and str(symbol).upper().startswith('BTC'):
                    fused_score *= 1 + 0.1 * dom
                else:
                    fused_score *= 1 - 0.1 * dom
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
                fused_score *= 1 + 0.05 * hot
        oi_overheat = False
        if open_interest is not None:
            oi_chg = open_interest.get('oi_chg')
            if oi_chg is not None:
                fused_score *= 1 + 0.1 * oi_chg
                if oi_chg > 0.5:
                    fused_score *= 0.8
                    oi_overheat = True

        # ===== 7. 如果 fused_score 为 NaN，直接返回无信号 =====
        if fused_score is None or (isinstance(fused_score, float) and np.isnan(fused_score)):
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

        # ===== 8. 把 fused_score 入历史，用于动态阈值计算 =====
        self.history_scores.append(fused_score)

        # ===== 9. 极端行情保护 =====
        if all_scores_list is not None:
            crowding_dir = self.crowding_protection(np.sign(all_scores_list))
            if (crowding_dir != 0) and (np.sign(fused_score) == crowding_dir):
                fused_score *= 0.5

        # ===== 10. 准备 details，用于回测与调试 =====
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
        )

        if abs(fused_score) < th:
            return {
                'signal': 0,
                'score': fused_score,
                'position_size': 0.0,
                'take_profit': None,
                'stop_loss': None,
                'details': details
            }

        # ===== 12. 多级仓位逻辑（替代 calculate_position_size） =====
        abs_score = abs(fused_score)
        if abs_score > 0.8:
            pos_size = 1.0
        elif abs_score > 0.5:
            pos_size = 0.6
        elif abs_score > 0.3:
            pos_size = 0.3
        else:
            pos_size = 0.1

        if oi_overheat:
            pos_size *= 0.7

        # ===== 13. 止盈止损计算：使用 ATR 动态设置 =====
        price = features_1h.get('close', 0)
        if raw_features_4h is not None and 'atr_pct_4h' in raw_features_4h:
            atr_pct_4h = raw_features_4h['atr_pct_4h']
        else:
            atr_pct_4h = features_4h.get('atr_pct_4h', 0)
        atr_abs = atr_pct_4h * price
        direction = int(np.sign(fused_score)) if fused_score != 0 else 1
        take_profit, stop_loss = self.compute_tp_sl(price, atr_abs, direction)

        # ===== 14. 最终返回 =====
        direction = int(np.sign(fused_score))  # 1 表示做多，-1 表示做空
        return {
            'signal': direction,
            'score': fused_score,
            'position_size': pos_size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'details': details
        }

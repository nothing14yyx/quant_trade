import joblib
import numpy as np
import pandas as pd
from collections import Counter, deque
pd.set_option('future.no_silent_downcasting', True)

class RobustSignalGenerator:
    """
    多周期AI+多因子+动态阈值+极端行情防护 融合信号生成器（调研增强版）
    """

    def __init__(self, model_paths, *, feature_cols_1h, feature_cols_4h, feature_cols_d1, history_window=500):
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
            'ai': 0.2,
            'trend': 0.2,
            'momentum': 0.2,
            'volatility': 0.2,
            'volume': 0.1,
            'sentiment': 0.05,
            'funding': 0.05
        }

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
        proba_down = lgb_model.predict_proba(X_df)[0][0]
        return self._score_from_proba(proba_up) - self._score_from_proba(proba_down)

    # robust_signal_generator.py

    def get_factor_scores(self, features: dict, period: str) -> dict:
        """
        输入：
          - features: 单周期特征字典（如 {'ema_diff_1h': 0.12, 'boll_perc_1h': 0.45, ...}）
          - period:   "1h" / "4h" / "d1"
        输出：一个 dict，包含6个子因子得分。
        """

        def safe(key: str, default=0):
            """如果指定 key 不存在或 value 为 None，就返回 default，否则返回实际值。"""
            v = features.get(key, default)
            return default if v is None else v

        return {
            # —— 趋势 因子 ——
            # 1) ema_diff     → ema_diff_{period}
            # 2) boll_perc    → (close - lower_bb) / (upper_bb - lower_bb)
            # 3) supertrend   → supertrend_dir_{period}
            # 4) adx_delta    → adx_delta_{period}
            # 5) bull/bear    → bull_streak_{period} - bear_streak_{period}
            'trend': (
                    np.tanh(safe(f'ema_diff_{period}', 0) * 5)
                    + 2 * (safe(f'boll_perc_{period}', 0.5) - 0.5)
                    + safe(f'supertrend_dir_{period}', 0)
                    + np.tanh(safe(f'adx_delta_{period}', 0) / 10)
                    + np.tanh((safe(f'bull_streak_{period}', 0) - safe(f'bear_streak_{period}', 0)) / 3)
            ),

            # —— 动量 因子 ——
            # 1) rsi          → rsi_{period}
            # 2) willr        → willr_{period}
            # 3) macd_hist    → macd_hist_{period}
            # 4) rsi_slope    → rsi_slope_{period}
            # 5) mfi          → mfi_{period}
            'momentum': (
                    (safe(f'rsi_{period}', 50) - 50) / 50
                    + safe(f'willr_{period}', -50) / 50
                    + np.tanh(safe(f'macd_hist_{period}', 0) * 5)
                    + np.tanh(safe(f'rsi_slope_{period}', 0) * 10)
                    + (safe(f'mfi_{period}', 50) - 50) / 50
            ),

            # —— 波动 因子 ——
            # 1) atr_pct      → atr_pct_{period}（% 带入 tanh 拉伸）
            # 2) bb_width     → bb_width_{period}
            # 3) donchian_delta → donchian_delta_{period}
            'volatility': (
                    np.tanh(safe(f'atr_pct_{period}', 0) * 8)
                    + np.tanh(safe(f'bb_width_{period}', 0) * 2)
                    + np.tanh(safe(f'donchian_delta_{period}', 0) * 5)
            ),

            # —— 成交量 因子 ——
            # 1) vol_ma_ratio → vol_ma_ratio_{period}
            # 2) obv_delta    → obv_delta_{period}
            # 3) vol_roc      → vol_roc_{period}
            # 4) rsi_mul_vol_ma_ratio → rsi_mul_vol_ma_ratio_{period}
            'volume': (
                    np.tanh(safe(f'vol_ma_ratio_{period}', 0))
                    + np.tanh(safe(f'obv_delta_{period}', 0) / 1e5)
                    + np.tanh(safe(f'vol_roc_{period}', 0) / 5)
                    + np.tanh(safe(f'rsi_mul_vol_ma_ratio_{period}', 0) / 100)
            ),

            # —— 情绪 因子 ——
            # 固定用日线情绪：fg_index_d1
            'sentiment': (safe('fg_index_d1', 50) - 50) / 50,

            # —— 资金费率 因子 ——
            # funding_rate_{period}
            'funding': np.tanh(safe(f'funding_rate_{period}', 0) * 100),
        }

    def dynamic_weight_update(self):
        # IC驱动动态权重分配（可扩展为每因子滑窗IC）
        ic_arr = np.array([max(0, v) for v in self.ic_scores.values()])
        base_arr = np.array([self.base_weights[k] for k in self.ic_scores.keys()])

        combined = ic_arr * base_arr
        if combined.sum() > 0:
            w = combined / combined.sum()
        else:
            w = base_arr / base_arr.sum()

        return dict(zip(self.ic_scores.keys(), w))

    def dynamic_threshold(self, atr, adx, funding=0, base=0.12, min_thres=0.06, max_thres=0.25):
        # 波动/趋势/资金费率 动态加权 + 分位阈值
        thres = base + min(0.08, abs(atr) * 3) + min(0.08, max(adx - 20, 0) * 0.004) + min(0.05, abs(funding) * 5)
        # 分位阈值补充防“弱信号出手”
        if len(self.history_scores) > 100:
            quantile_th = np.quantile(np.abs(self.history_scores), 0.92)
            thres = max(thres, quantile_th)
        return max(min_thres, min(max_thres, thres))

    def combine_score(self, ai_score, factor_scores, weights=None):
        # 动态排序赋权 + 中位融合（调研建议）
        if weights is None:
            weights = self.base_weights
        raw_scores = [
            ai_score,
            factor_scores['trend'],
            factor_scores['momentum'],
            factor_scores['volatility'],
            factor_scores['volume'],
            factor_scores['sentiment'],
            factor_scores['funding']
        ]
        # 排序赋权（绝对值越大权重越高）
        abs_idx = np.argsort(-np.abs(raw_scores))
        sorted_scores = np.array(raw_scores)[abs_idx]

        weight_values = np.array([weights[k] for k in weights])
        sorted_weights = np.sort(weight_values)[::-1]

        fused_score = np.dot(sorted_scores, sorted_weights)
        # 可选：中位融合
        # fused_score = np.median(raw_scores)
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
        输出：
            一个 dict，包含 'signal'、'score'、'position_size'、'take_profit'、'stop_loss' 和 'details'
        说明：若传入 raw_features_*，则多因子计算与动态阈值、止盈止损均使用原始数据，
              标准化后的 features_* 仅用于 AI 模型预测。
        """

        # ===== 1. 计算 AI 部分的分数（映射到 [-1, 1]） =====
        ai_scores = {}
        for p, feats in [('1h', features_1h), ('4h', features_4h), ('d1', features_d1)]:
            score_up = self.get_ai_score(feats, self.models[p]['up'])
            score_down = self.get_ai_score(feats, self.models[p]['down'])
            ai_scores[p] = 0.5 * (score_up - score_down)

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

        if consensus_all:
            fused_score = 0.7 * score_1h + 0.2 * score_4h + 0.1 * score_d1
            if strong_confirm:
                fused_score *= 1.15
        elif consensus_14:
            fused_score = 0.75 * score_1h + 0.25 * score_4h
            if strong_confirm:
                fused_score *= 1.10
        else:
            fused_score = score_1h

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
            'consensus_14': consensus_14, 'consensus_all': consensus_all
        }

        # ===== 11. 动态阈值过滤，调用已有 dynamic_threshold =====
        raw_f1h = raw_features_1h or features_1h
        atr_1h = raw_f1h.get('atr_pct_1h', features_1h.get('atr_pct_1h', 0))
        adx_1h = raw_f1h.get('adx_1h', features_1h.get('adx_1h', 0))
        funding_1h = raw_f1h.get('funding_rate_1h', features_1h.get('funding_rate_1h', 0)) or 0
        th = self.dynamic_threshold(atr_1h, adx_1h, funding_1h)

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

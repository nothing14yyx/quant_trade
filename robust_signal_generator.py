import joblib
import numpy as np
import pandas as pd
from collections import Counter, deque

class RobustSignalGenerator:
    """
    多周期AI+多因子+动态阈值+极端行情防护 融合信号生成器（调研增强版）
    """

    def __init__(self, model_paths, feature_cols_1h, feature_cols_4h, feature_cols_d1, history_window=500):
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
            'ai': 0.42,
            'trend': 0.14,
            'momentum': 0.12,
            'volatility': 0.08,
            'volume': 0.13,
            'sentiment': 0.06,
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

    # >>>>> 修改：改写 get_ai_score，让它自动从 self.models[...]["features"] 中取“训练时列名”
    def get_ai_score(self, features, model_dict):
        """
        features: 当前币种的特征字典（key: 列名，value: 数值）
        model_dict: 形如 {"model": LGBMClassifier, "features": [...训练时列...]}

        返回：AI 置信度映射到 [-1,1] 的综合得分
        """
        # 1. 取出模型和它训练时用的列名
        lgb_model   = model_dict["model"]
        train_cols  = model_dict["features"]  # 训练时的那份列名列表，比如共 14 列

        # 2. 按照 train_cols 的顺序，逐列从 features 字典里取值，构造 DataFrame
        #    如果某列在 features 里缺失，就用 0 填补
        row_data = {col: [features.get(col, 0)] for col in train_cols}
        X_df = pd.DataFrame(row_data)

        # 3. 预测“上涨概率”、“下跌概率”，映射到 [-1,1] 并相减
        proba_up   = lgb_model.predict_proba(X_df)[0][1]
        proba_down = lgb_model.predict_proba(X_df)[0][0]  # 下跌概率是第 0 列，或 1 - up
        # 如果你在训时是二分类且 predict_proba 返回 [p_down, p_up]，那么 proba_down = [0][0] 即可。
        # 也可以直接写 proba_down = 1 - proba_up

        return self._score_from_proba(proba_up) - self._score_from_proba(proba_down)

    def get_factor_scores(self, features, period):
        def safe(key, default=0):
            return features.get(key, default)

        return {
            'trend': np.tanh(safe(f'ema_diff_{period}', 0) * 5) + 2 * (safe(f'boll_perc_{period}', 0.5) - 0.5) + safe(f'supertrend_dir_{period}', 0),
            'momentum': (safe(f'rsi_{period}', 50) - 50) / 50 + safe(f'willr_{period}', -50) / 50 + np.tanh(safe(f'macd_hist_{period}', 0) * 5),
            'volatility': np.tanh(safe(f'atr_pct_{period}', 0) * 8),
            'volume': np.tanh(safe(f'vol_ma_ratio_{period}', 0)) + np.tanh(safe(f'obv_delta_{period}', 0) / 1e5),
            'sentiment': (safe('fg_index_d1', 50) - 50) / 50,
            'funding': np.tanh(safe(f'funding_rate_{period}', 0) * 100),
        }

    def dynamic_weight_update(self):
        # IC驱动动态权重分配（可扩展为每因子滑窗IC）
        ic_arr = np.array([max(0, v) for v in self.ic_scores.values()])
        if ic_arr.sum() > 0:
            w = ic_arr / ic_arr.sum()
        else:
            w = np.ones_like(ic_arr) / len(ic_arr)
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
        sorted_weights = np.array([weights[k] for k in weights])
        sorted_weights = sorted_weights[abs_idx]
        sorted_scores = np.array(raw_scores)[abs_idx]
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

    def generate_signal(self, features_1h, features_4h, features_d1, all_scores_list=None):
        """
        输入：三个周期特征 dict，all_scores_list: 当前所有币种的分数（用于极端行情过滤）
        输出：signal, score, position_size, details
        """
        # >>>>> 修改点：把 get_ai_score 的调用全部改为只传 model_dict，不再传 feature_cols_xx
        ai_scores = {
            '1h': self.get_ai_score(features_1h, self.models['1h']['up'])   -
                   self.get_ai_score(features_1h, self.models['1h']['down']),
            '4h': self.get_ai_score(features_4h, self.models['4h']['up'])   -
                   self.get_ai_score(features_4h, self.models['4h']['down']),
            'd1': self.get_ai_score(features_d1, self.models['d1']['up'])   -
                   self.get_ai_score(features_d1, self.models['d1']['down'])
        }

        fs = {
            '1h': self.get_factor_scores(features_1h, '1h'),
            '4h': self.get_factor_scores(features_4h, '4h'),
            'd1': self.get_factor_scores(features_d1, 'd1')
        }
        weights = self.dynamic_weight_update()
        score_1h = self.combine_score(ai_scores['1h'], fs['1h'], weights)
        score_4h = self.combine_score(ai_scores['4h'], fs['4h'], weights)
        score_d1 = self.combine_score(ai_scores['d1'], fs['d1'], weights)

        # 多指标一致性门控（调研增强）
        strong_confirm = (
            (fs['4h']['trend'] > 0 and fs['4h']['momentum'] > 0 and fs['4h']['volatility'] > 0 and score_4h > 0) or
            (fs['4h']['trend'] < 0 and fs['4h']['momentum'] < 0 and fs['4h']['volatility'] < 0 and score_4h < 0)
        )
        consensus_dir = self.consensus_check(score_1h, score_4h, score_d1)
        if consensus_dir != 0:
            # 共振方向强时主用4h分数
            fused_score = score_4h
            if strong_confirm:
                fused_score *= 1.15  # 强信号加权
        else:
            fused_score = 0.5 * score_4h  # 分歧时弱化

        # 分数历史入队，给动态分位阈值用
        self.history_scores.append(fused_score)

        # 阈值
        atr_4h = features_4h.get('atr_pct_4h', 0)
        adx_4h = features_4h.get('adx_4h', 25)
        funding_4h = features_4h.get('funding_rate_4h', 0)
        th = self.dynamic_threshold(atr_4h, adx_4h, funding_4h)

        # 极端行情过滤
        if all_scores_list is not None:
            crowding_dir = self.crowding_protection(np.sign(all_scores_list))
            if crowding_dir != 0 and np.sign(fused_score) == crowding_dir:
                fused_score *= 0.5  # 自动降权

        details = {
            'ai_1h': ai_scores['1h'], 'ai_4h': ai_scores['4h'], 'ai_d1': ai_scores['d1'],
            'factors_1h': fs['1h'], 'factors_4h': fs['4h'], 'factors_d1': fs['d1'],
            'score_1h': score_1h, 'score_4h': score_4h, 'score_d1': score_d1,
            'atr_4h': atr_4h, 'adx_4h': adx_4h, 'funding_4h': funding_4h, 'threshold': th,
            'strong_confirm': strong_confirm, 'consensus_dir': consensus_dir
        }

        # 动态阈值过滤弱信号
        if abs(fused_score) < th:
            return {
                'signal': 0,
                'score': fused_score,
                'position_size': 0.0,
                'details': details
            }
        # 多级仓位（调研建议，非线性映射）
        abs_score = abs(fused_score)
        if abs_score > 0.8:
            pos_size = 1.0
        elif abs_score > 0.5:
            pos_size = 0.6
        elif abs_score > 0.3:
            pos_size = 0.3
        else:
            pos_size = 0.1

        direction = 1 if fused_score > 0 else -1
        return {
            'signal': direction,
            'score': fused_score,
            'position_size': pos_size,
            'details': details
        }

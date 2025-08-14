"""AI 模型推理工具函数."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
from scipy.special import inv_boxcox


def _hash_features(feats: Mapping[str, float | int | None]) -> int:
    """将特征映射转换为可哈希的整数."""
    return hash(tuple(sorted(feats.items())))


def get_period_ai_scores(
    predictor: Any | None,
    period_features: Mapping[str, Mapping[str, float | int | None]],
    models: Mapping[str, Mapping[str, Any]],
    calibrators: Mapping[str, Mapping[str, Any]] | None = None,
    cache: Any | None = None,
) -> dict[str, float]:
    """根据各周期模型计算 AI 得分.

    参数:
        predictor: 具备 ``get_ai_score`` 与 ``get_ai_score_cls`` 方法的预测器.
        period_features: 周期到特征映射.
        models: 周期到模型字典的映射.
        calibrators: 周期到校准器的映射.

    返回:
        每个周期的 AI 得分字典, 若无模型或预测器则得分为 0.
    """

    calibrators = calibrators or {}
    ai_scores: dict[str, float] = {}
    if predictor is None:
        for p in period_features:
            ai_scores[p] = 0.0
        return ai_scores

    for period, feats in period_features.items():
        models_p = models.get(period, {})
        if not models_p:
            ai_scores[period] = 0.0
            continue

        key = (period, _hash_features(feats))
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                ai_scores[period] = cached
                continue

        if "cls" in models_p and "up" not in models_p:
            score_val = predictor.get_ai_score_cls(feats, models_p["cls"])
        else:
            cal_up = calibrators.get(period, {}).get("up")
            cal_down = calibrators.get(period, {}).get("down")
            if cal_up is None and cal_down is None:
                score_val = predictor.get_ai_score(
                    feats, models_p.get("up", {}), models_p.get("down", {})
                )
            else:
                score_val = predictor.get_ai_score(
                    feats,
                    models_p.get("up", {}),
                    models_p.get("down", {}),
                    cal_up,
                    cal_down,
                )
        ai_scores[period] = score_val
        if cache is not None:
            cache.set(key, score_val)
    for p in period_features:
        ai_scores.setdefault(p, 0.0)
    return ai_scores


def get_reg_predictions(
    predictor: Any | None,
    period_features: Mapping[str, Mapping[str, float | int | None]],
    models: Mapping[str, Mapping[str, Any]],
    rise_transform: str = "none",
    boxcox_lambda: Mapping[str, float] | None = None,
) -> Tuple[dict[str, float | None], dict[str, float | None], dict[str, float | None]]:
    """获取回归预测结果.

    返回三个字典分别代表波动率、涨幅与回撤预测.
    若缺少预测器或模型则相应值为 ``None``。
    """

    boxcox_lambda = boxcox_lambda or {}
    vol_preds: dict[str, float | None] = {}
    rise_preds: dict[str, float | None] = {}
    drawdown_preds: dict[str, float | None] = {}

    if predictor is None:
        for p in period_features:
            vol_preds[p] = None
            rise_preds[p] = None
            drawdown_preds[p] = None
        return vol_preds, rise_preds, drawdown_preds

    for period, feats in period_features.items():
        models_p = models.get(period, {})
        if not models_p:
            vol_preds.setdefault(period, None)
            rise_preds.setdefault(period, None)
            drawdown_preds.setdefault(period, None)
            continue
        if "vol" in models_p:
            vol_preds[period] = predictor.get_vol_prediction(feats, models_p["vol"])
        if "rise" in models_p:
            pred = predictor.get_reg_prediction(feats, models_p["rise"])
            if rise_transform == "log":
                pred = float(np.expm1(pred))
            elif rise_transform == "boxcox":
                lmbda = boxcox_lambda.get(period)
                if lmbda is not None:
                    pred = float(inv_boxcox(pred, lmbda) - 1.0)
            rise_preds[period] = pred
        if "drawdown" in models_p:
            drawdown_preds[period] = predictor.get_reg_prediction(
                feats, models_p["drawdown"]
            )
        vol_preds.setdefault(period, None)
        rise_preds.setdefault(period, None)
        drawdown_preds.setdefault(period, None)

    return vol_preds, rise_preds, drawdown_preds

"""AI 模型推理工具函数."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

import numpy as np
from scipy.special import inv_boxcox


def _hash_features(feats: Mapping[str, float | int | None]) -> int:
    """将特征映射转换为可哈希的整数."""
    return hash(tuple(sorted(feats.items())))


def compute_ai_scores_batch(
    predictor: Any | None,
    period_features_batch,
    models: Mapping[str, Mapping[str, Any]],
    calibrators: Mapping[str, Mapping[str, Any]] | None = None,
    cache: Any | None = None,
) -> list[dict[str, float]]:
    """批量计算各周期 AI 分数.

    ``period_features_batch`` 可以是 ``{period: list[Mapping]}``，也可以是形如
    ``numpy.ndarray`` 的二维数组（仅针对单个周期）。返回值为每条样本对应
    的 ``{period: score}`` 字典列表。
    """

    calibrators = calibrators or {}

    if predictor is None:
        if isinstance(period_features_batch, Mapping):
            n = len(next(iter(period_features_batch.values())))
            return [
                {p: 0.0 for p in period_features_batch}
                for _ in range(n)
            ]
        arr = np.asarray(period_features_batch)
        return [
            {next(iter(models.keys()), "1h"): 0.0} for _ in range(len(arr))
        ]

    if isinstance(period_features_batch, Mapping):
        periods = list(period_features_batch.keys())
        n = len(next(iter(period_features_batch.values())))
        results: list[dict[str, float]] = [dict() for _ in range(n)]
        for period in periods:
            feats_list = period_features_batch[period]
            models_p = models.get(period, {})
            if not models_p:
                for r in results:
                    r[period] = 0.0
                continue
            if "cls" in models_p and "up" not in models_p:
                scores = predictor.get_ai_score_cls(feats_list, models_p["cls"])
            else:
                cal_up = calibrators.get(period, {}).get("up")
                cal_down = calibrators.get(period, {}).get("down")
                scores = predictor.get_ai_score(
                    feats_list,
                    models_p.get("up", {}),
                    models_p.get("down", {}),
                    cal_up,
                    cal_down,
                )
            scores_arr = np.asarray(scores, dtype=float).reshape(-1)
            for i, sc in enumerate(scores_arr):
                results[i][period] = float(sc)
                if cache is not None and isinstance(feats_list, list) and feats_list:
                    f = feats_list[i] if feats_list[i] is not None else {}
                    key = (period, _hash_features(f))
                    cache.set(key, float(sc))
        return results

    # period_features_batch is array for a single period
    arr = np.asarray(period_features_batch)
    n = arr.shape[0]
    period = next(iter(models.keys()), "1h")
    models_p = models.get(period, {})
    if not models_p:
        return [{period: 0.0} for _ in range(n)]
    if "cls" in models_p and "up" not in models_p:
        scores = predictor.get_ai_score_cls(arr, models_p["cls"])
    else:
        cal_up = calibrators.get(period, {}).get("up")
        cal_down = calibrators.get(period, {}).get("down")
        scores = predictor.get_ai_score(
            arr,
            models_p.get("up", {}),
            models_p.get("down", {}),
            cal_up,
            cal_down,
        )
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    return [{period: float(s)} for s in scores_arr]


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

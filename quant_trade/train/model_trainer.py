from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class PurgedKFold:
    """K-fold cross-validator for time series with an *embargo* gap.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    embargo_td : pd.Timedelta, default=pd.Timedelta(0)
        Length of the time gap to exclude before and after each validation
        interval from the training data.
    """

    n_splits: int = 5
    embargo_td: pd.Timedelta = pd.Timedelta(0)

    def split(
        self,
        X: Iterable,
        y: Iterable | None = None,
        times: pd.Series | None = None,
        price: pd.Series | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into train and validation sets.

        Parameters
        ----------
        X, y : array-like
            Training data. Only the length of ``X`` is inspected.
        times : pd.Series
            Timestamps aligned with ``X`` used to determine the validation
            windows and the embargo regions.
        price : pd.Series, optional
            Price series aligned with ``X``. If provided, its length must
            match ``X`` and missing values will raise an error.
        """

        if times is None:
            raise ValueError("times must be provided")

        times = pd.Series(times).reset_index(drop=True)
        n_samples = len(times)
        if len(X) != n_samples:
            raise ValueError("times and X have different lengths")
        if price is not None:
            price = pd.Series(price).reset_index(drop=True)
            if len(price) != n_samples:
                raise ValueError("price and X have different lengths")
            if price.isna().any():
                raise ValueError("price contains missing values")
        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError("n_splits must be between 2 and n_samples")

        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for i, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            val_start_time = times.iloc[start]
            val_end_time = times.iloc[stop - 1]

            embargo_start = val_start_time - self.embargo_td
            embargo_end = val_end_time + self.embargo_td
            train_mask = (times < embargo_start) | (times > embargo_end)
            train_indices = indices[train_mask.to_numpy()]

            if len(train_indices) == 0:
                if self.embargo_td > pd.Timedelta(0):
                    warnings.warn(
                        f"Fold {i} has no training samples with embargo_td="
                        f"{self.embargo_td}. Reducing embargo_td to 0.",
                        RuntimeWarning,
                    )
                    train_mask = (times < val_start_time) | (times > val_end_time)
                    train_indices = indices[train_mask.to_numpy()]
                if len(train_indices) == 0:
                    warnings.warn(
                        f"Fold {i} still has no training samples; skipping this fold. "
                        "Consider reducing n_splits or providing more data.",
                        RuntimeWarning,
                    )
                    current = stop
                    continue

            if price is not None:
                # Retrieve the price series for the validation window
                test_price = price.iloc[test_indices]
                if test_price.isna().any():
                    raise ValueError(
                        "price contains missing values in validation set"
                    )

            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def train_with_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    times: pd.Series,
    *,
    n_splits: int = 5,
    save_path: str | Path | None = None,
) -> dict:
    """使用 ``PurgedKFold`` 执行交叉验证并返回报告。

    Parameters
    ----------
    model : sklearn-like estimator
        需要实现 ``fit`` 与 ``predict_proba`` 方法。
    X : pd.DataFrame
        特征数据。
    y : pd.Series
        二分类标签。
    times : pd.Series
        与 ``X`` 对齐的时间索引，用于生成折叠。
    n_splits : int, default=5
        交叉验证折数。
    save_path : str or Path, optional
        若提供，则把 ``report`` 以 JSON 格式写入该路径。

    Returns
    -------
    dict
        ``{"cv": {...}, "bt": {...}}``，分别为交叉验证平均指标
        与全量回测指标。
    """

    pkf = PurgedKFold(n_splits=n_splits)
    cv: dict[str, list[float]] = {
        "AUC": [],
        "LogLoss": [],
        "Precision": [],
        "Recall": [],
    }

    for tr, va in pkf.split(X, y, times=times):
        clf = clone(model)
        clf.fit(X.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X.iloc[va])
        y_va = y.iloc[va]

        cv["AUC"].append(roc_auc_score(y_va, proba[:, 1]))
        cv["LogLoss"].append(log_loss(y_va, proba))
        pred = (proba[:, 1] >= 0.5).astype(int)
        cv["Precision"].append(
            precision_score(y_va, pred, zero_division=0)
        )
        cv["Recall"].append(recall_score(y_va, pred, zero_division=0))

    cv_avg = {k: float(np.mean(v)) for k, v in cv.items()}

    model.fit(X, y)
    proba_bt = model.predict_proba(X)
    pred_bt = (proba_bt[:, 1] >= 0.5).astype(int)
    bt = {
        "AUC": float(roc_auc_score(y, proba_bt[:, 1])),
        "LogLoss": float(log_loss(y, proba_bt)),
        "Precision": float(precision_score(y, pred_bt, zero_division=0)),
        "Recall": float(recall_score(y, pred_bt, zero_division=0)),
    }

    report = {"cv": cv_avg, "bt": bt}
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report

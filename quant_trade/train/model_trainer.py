from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


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
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into train and validation sets.

        Parameters
        ----------
        X, y : array-like
            Training data. Only the length of ``X`` is inspected.
        times : pd.Series
            Timestamps aligned with ``X`` used to determine the validation
            windows and the embargo regions.
        """

        if times is None:
            raise ValueError("times must be provided")

        times = pd.Series(times).reset_index(drop=True)
        n_samples = len(times)
        if len(X) != n_samples:
            raise ValueError("times and X have different lengths")
        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError("n_splits must be between 2 and n_samples")

        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            val_start_time = times.iloc[start]
            val_end_time = times.iloc[stop - 1]

            embargo_start = val_start_time - self.embargo_td
            embargo_end = val_end_time + self.embargo_td
            train_mask = (times < embargo_start) | (times > embargo_end)
            train_indices = indices[train_mask.to_numpy()]

            yield train_indices, test_indices
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

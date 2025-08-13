"""Purged time series cross-validator with embargo support.

This implementation follows the idea of López de Prado's purged
k-fold cross-validation. Compared to :class:`~sklearn.model_selection.TimeSeriesSplit`,
it removes a number of samples immediately preceding the validation set
(the *embargo* period) from the training set to mitigate label leakage
caused by overlapping horizons.

Example
-------
>>> import numpy as np
>>> from quant_trade.utils.purged_split import PurgedTimeSeriesSplit
>>> pts = PurgedTimeSeriesSplit(n_splits=3, embargo=2)
>>> X = np.arange(12)
>>> for tr, te in pts.split(X):
...     print(tr, te)
...
[0 1 2 3 4] [6 7 8]
[0 1 2 3 4 5 6] [8 9 10]
[0 1 2 3 4 5 6 7] [10 11]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Iterable, Tuple

import numpy as np


@dataclass
class PurgedTimeSeriesSplit:
    """Time series cross-validator with an *embargo* gap.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    max_train_size : int or None, default=None
        Maximum size for a single training set.
    test_size : int or None, default=None
        Size of each validation fold. If ``None`` (default), it is set to
        ``n_samples // (n_splits + 1)`` similar to
        :class:`~sklearn.model_selection.TimeSeriesSplit`.
    embargo : int, default=0
        Number of samples to skip between the end of the training set and
        the start of the validation set.
    """

    n_splits: int = 5
    max_train_size: int | None = None
    test_size: int | None = None
    embargo: int = 0

    def split(
        self,
        X: Iterable,
        y: Iterable | None = None,
        groups: Iterable | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and validation set."""

        n_samples = len(X)
        if self.n_splits <= 0:
            raise ValueError("n_splits must be at least 1")
        if self.n_splits >= n_samples:
            raise ValueError(
                "n_splits=%d cannot be >= n_samples=%d" % (self.n_splits, n_samples)
            )

        test_size = self.test_size
        if test_size is None:
            test_size = n_samples // (self.n_splits + 1)
            if test_size == 0:
                raise ValueError("test_size becomes 0 with current n_splits")
        indices = np.arange(n_samples)

        test_starts = range(
            n_samples - test_size * self.n_splits, n_samples, test_size
        )
        for test_start in test_starts:
            train_end = test_start - self.embargo
            if train_end <= 0:
                train_indices = np.empty(0, dtype=int)
            else:
                train_indices = indices[:train_end]
                if self.max_train_size is not None and len(train_indices) > self.max_train_size:
                    train_indices = train_indices[-self.max_train_size :]
            test_indices = indices[test_start : test_start + test_size]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

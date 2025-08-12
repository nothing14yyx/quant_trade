"""Voting model for probabilistic vote aggregation.

This module provides utilities to train and use a simple classifier that maps
several directional signals (``ai_dir``, ``short_mom_dir`` …) to a probability
score.  The model can be trained with logistic regression or GBDT and the
trained artifact is stored under ``quant_trade/models/voting_model.pkl`` by
default.

Training script example::

    python -m quant_trade.signal.voting_model --data training.csv --model logistic

The training CSV must contain the following columns::

    ai_dir, short_mom_dir, vol_breakout_dir, trend_dir, confirm_dir, ob_dir, label

``label`` should be ``1`` when a long vote is desired, and ``0`` for short.  No
additional feature engineering is performed; the directional features are used
as-is.  The module records all steps so that model training can be reproduced
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# 默认模型保存路径
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "voting_model.pkl"


@dataclass
class VotingModel:
    """Wrapper around a scikit-learn classifier."""

    model: object

    FEATURE_COLS = [
        "ai_dir",
        "short_mom_dir",
        "vol_breakout_dir",
        "trend_dir",
        "confirm_dir",
        "ob_dir",
    ]

    @classmethod
    def train(
        cls,
        data: pd.DataFrame,
        *,
        model_type: str = "logistic",
        **kwargs,
    ) -> "VotingModel":
        """Train a voting model.

        Parameters
        ----------
        data:
            DataFrame containing feature columns and a ``label`` column where
            ``1`` denotes long and ``0`` denotes short.
        model_type:
            Either ``"logistic"`` or ``"gbdt"``.
        kwargs:
            Extra keyword arguments passed to the scikit-learn estimator.
        """

        X = data[cls.FEATURE_COLS].values
        y = data["label"].values
        if model_type == "gbdt":
            est = GradientBoostingClassifier(**kwargs)
        else:
            est = LogisticRegression(**kwargs)
        est.fit(X, y)
        return cls(est)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        """Persist model to ``path``."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "VotingModel":
        """Load a trained model from ``path``."""

        model = joblib.load(path)
        return cls(model)

    def predict_proba(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Return probability of positive class for ``X``."""

        X = np.asarray(list(X), dtype=float)
        return self.model.predict_proba(X)


# 简单的缓存, 避免重复加载模型
_model_cache: VotingModel | None = None


def load_cached_model(path: Path = DEFAULT_MODEL_PATH) -> VotingModel:
    """Load and cache the model instance."""

    global _model_cache
    if _model_cache is None:
        _model_cache = VotingModel.load(path)
    return _model_cache


def safe_load(path: Path = DEFAULT_MODEL_PATH) -> VotingModel | None:
    """Safely load a model; return ``None`` if loading fails."""

    try:
        model = joblib.load(path)
    except Exception:
        return None
    return VotingModel(model)


def _main():  # pragma: no cover - simple CLI
    import argparse

    parser = argparse.ArgumentParser(description="Train voting model")
    parser.add_argument("--data", required=True, help="CSV 文件路径, 包含特征与 label")
    parser.add_argument(
        "--model",
        choices=["logistic", "gbdt"],
        default="logistic",
        help="训练模型类型",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_MODEL_PATH),
        help="输出模型文件路径",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    vm = VotingModel.train(df, model_type=args.model)
    vm.save(Path(args.out))
    print(f"model saved to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()

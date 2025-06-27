import numpy as np


def soft_clip(x, k=8.0):
    """Smooth saturate to (-k, k) using tanh."""
    return np.tanh(np.asarray(x, dtype=float) / k) * k

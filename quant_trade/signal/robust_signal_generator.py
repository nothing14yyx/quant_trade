from __future__ import annotations
import numpy as np, pandas as pd

def compute_dynamic_threshold(history_scores,
                              *args,
                              base_q: float = 0.78,
                              window: int = 150,
                              ewm_alpha: float = 0.20,
                              clamp: tuple[float, float] = (0.45, 0.90),
                              atr: float | None = None,
                              adx: float | None = None,
                              vix_proxy: float | None = None,
                              **legacy_kwargs) -> float:
    """Compute adaptive quantile threshold.

    Supports legacy positional or keyword arguments:
    ``compute_dynamic_threshold(scores, window, quantile)`` and keyword
    aliases ``quantile`` -> ``base_q`` and ``th_window`` -> ``window``.
    """
    if args:
        if len(args) >= 1:
            window = args[0]
        if len(args) >= 2:
            base_q = args[1]
    if "quantile" in legacy_kwargs and legacy_kwargs["quantile"] is not None:
        base_q = legacy_kwargs.pop("quantile")
    if "th_window" in legacy_kwargs and legacy_kwargs["th_window"] is not None:
        window = legacy_kwargs.pop("th_window")

    hist_list = list(history_scores)[-int(window):]
    hist = np.asarray(hist_list, dtype=float)
    if hist.size == 0:
        return float(np.clip(base_q, *clamp))
    q = base_q if hist.size < max(50, int(window * 0.3)) else float(np.nanquantile(hist, base_q))
    smoothed = pd.Series(list(history_scores), dtype="float64").ewm(alpha=ewm_alpha, adjust=False).mean().iloc[-1]

    def _z(x, m=np.nanmedian(hist), s=(np.nanstd(hist) + 1e-6)):
        if x is None or np.isnan(x):
            return 0.0
        return np.tanh((x - m) / s)

    adj = 0.10 * _z(atr) + 0.08 * _z(adx) + 0.06 * _z(vix_proxy)
    q_eff = float(np.clip(q + 0.05 * np.tanh(smoothed) + adj, *clamp))
    return q_eff

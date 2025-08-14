from __future__ import annotations

"""Simple backtesting helpers."""

from typing import Optional, Tuple


def execute_signal(
    price_open_next: float,
    price_high_next: float,
    price_low_next: float,
    direction: int,
    *,
    slippage: float = 0.0,
) -> Tuple[Optional[float], float]:
    """Attempt to execute a pending order.

    Parameters
    ----------
    price_open_next : float
        The opening price of the next bar.
    price_high_next : float
        The high price of the next bar.
    price_low_next : float
        The low price of the next bar.
    direction : int
        Trade direction, ``1`` for buy and ``-1`` for sell.
    slippage : float, optional
        Fraction used to offset the opening price when placing the order.

    Returns
    -------
    tuple
        ``(fill_price, pnl)`` where ``fill_price`` is ``None`` if the order
        is not filled and ``pnl`` is ``0.0`` in that case.
    """
    if direction not in (1, -1):
        return None, 0.0

    # Determine the limit order price using an offset from the next open price.
    limit_price = (
        price_open_next * (1 - slippage)
        if direction == 1
        else price_open_next * (1 + slippage)
    )

    # Use strict inequalities so the order may remain unfilled.
    if direction == 1:
        if price_low_next < limit_price:
            return limit_price, 0.0
    else:
        if price_high_next > limit_price:
            return limit_price, 0.0

    return None, 0.0

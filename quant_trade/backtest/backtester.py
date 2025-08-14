from __future__ import annotations

"""Simple backtesting helpers."""

from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionPolicy:
    """Execution policy defining maker/taker fee rates."""

    maker_fee: float = 0.0
    taker_fee: float = 0.0

    def fee(self, price: float, qty: float, *, is_taker: bool) -> float:
        rate = self.taker_fee if is_taker else self.maker_fee
        return abs(price * qty) * rate


def execute_signal(
    price_open_next: float,
    price_high_next: float,
    price_low_next: float,
    direction: int,
    *,
    policy: ExecutionPolicy | None = None,
    slippage: float = 0.0,
    position_size: float = 1.0,
    is_taker: bool = False,
) -> Tuple[Optional[float], float]:
    """Attempt to execute a pending order and return fee if filled.

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
    policy : ExecutionPolicy, optional
        Fee policy used when computing transaction costs.
    slippage : float, optional
        Fraction used to offset the opening price when placing a limit order.
    position_size : float, optional
        Trade size used to compute fees.
    is_taker : bool, optional
        If ``True`` the order is executed immediately at ``price_open_next``.

    Returns
    -------
    tuple
        ``(fill_price, fee)`` where ``fill_price`` is ``None`` if the order
        is not filled and ``fee`` is ``0.0`` in that case.
    """
    if direction not in (1, -1):
        return None, 0.0

    policy = policy or ExecutionPolicy()

    if is_taker:
        fill_price = price_open_next
        fee = policy.fee(fill_price, position_size, is_taker=True)
        return fill_price, fee

    # Determine the limit order price using an offset from the next open price.
    limit_price = (
        price_open_next * (1 - slippage)
        if direction == 1
        else price_open_next * (1 + slippage)
    )

    # Use strict inequalities so the order may remain unfilled.
    if direction == 1:
        if price_low_next < limit_price:
            fee = policy.fee(limit_price, position_size, is_taker=False)
            return limit_price, fee
    else:
        if price_high_next > limit_price:
            fee = policy.fee(limit_price, position_size, is_taker=False)
            return limit_price, fee

    return None, 0.0

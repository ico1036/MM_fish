"""Avellaneda-Stoikov market making model — pure functions."""

import numpy as np


def reservation_price(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Reservation price (indifference price).

    r = s - q * gamma * sigma^2 * T

    Args:
        mid_price: Current mid price (s).
        inventory: Current inventory (q). Positive = long, negative = short.
        gamma: Risk aversion coefficient. Higher = more conservative.
        sigma: Volatility estimate.
        T: Remaining time fraction (0 to 1).

    Returns:
        The reservation price.
    """
    return mid_price - inventory * gamma * sigma**2 * T


def optimal_spread(
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> float:
    """
    Optimal spread around the reservation price.

    delta = gamma * sigma^2 * T + (2 / gamma) * ln(1 + gamma / k)

    Args:
        gamma: Risk aversion coefficient.
        sigma: Volatility estimate.
        T: Remaining time fraction.
        k: Order book liquidity parameter (order arrival intensity).

    Returns:
        The optimal spread (always positive).
    """
    return gamma * sigma**2 * T + (2.0 / gamma) * np.log(1.0 + gamma / k)


def compute_quotes(
    mid_price: float,
    inventory: float,
    gamma: float,
    sigma: float,
    T: float,
    k: float,
) -> tuple[float, float]:
    """
    Compute final bid/ask quotes.

    bid = r - delta / 2
    ask = r + delta / 2

    Args:
        mid_price: Current mid price.
        inventory: Current inventory.
        gamma: Risk aversion coefficient.
        sigma: Volatility estimate.
        T: Remaining time fraction.
        k: Order book liquidity parameter.

    Returns:
        (bid_price, ask_price) tuple.
    """
    r = reservation_price(mid_price, inventory, gamma, sigma, T)
    delta = optimal_spread(gamma, sigma, T, k)
    return r - delta / 2.0, r + delta / 2.0

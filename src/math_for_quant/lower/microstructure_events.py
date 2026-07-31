from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


def poisson_mle(interarrivals: list[float]) -> tuple[float, float]:
    if not interarrivals or any(not math.isfinite(x) or x <= 0.0 for x in interarrivals):
        raise ValueError("interarrival times must be positive and finite")
    total = float(sum(interarrivals))
    intensity = len(interarrivals) / total
    return intensity, len(interarrivals) * math.log(intensity) - intensity * total


def seasonally_adjusted_poisson_mle(
    interarrivals: list[float], multipliers: list[float]
) -> tuple[float, np.ndarray]:
    if len(interarrivals) != len(multipliers) or not interarrivals:
        raise ValueError("seasonal observations must be nonempty and aligned")
    waiting = np.asarray(interarrivals, dtype=float)
    seasonal = np.asarray(multipliers, dtype=float)
    if (
        np.any(~np.isfinite(waiting))
        or np.any(waiting <= 0.0)
        or np.any(~np.isfinite(seasonal))
        or np.any(seasonal <= 0.0)
    ):
        raise ValueError("seasonal waiting times and multipliers must be positive and finite")
    exposure = seasonal * waiting
    intensity = float(waiting.size / exposure.sum())
    return intensity, intensity * exposure


def hawkes_log_likelihood(
    times: np.ndarray,
    baseline: float,
    alpha: float,
    beta: float,
    *,
    horizon: float,
    initial_excitation: float = 0.0,
) -> float:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Hawkes event times must be strictly increasing")
    if not math.isfinite(horizon) or horizon < float(times[-1]) or horizon <= 0.0:
        raise ValueError("Hawkes horizon must be finite and include every event")
    if (
        not all(math.isfinite(x) for x in (baseline, alpha, beta, initial_excitation))
        or baseline <= 0.0
        or alpha < 0.0
        or beta <= 0.0
        or alpha / beta >= 1.0
        or initial_excitation < 0.0
    ):
        raise ValueError("Hawkes parameters violate positivity or stability")
    excitation = float(initial_excitation)
    previous = 0.0
    log_terms = []
    for time in times:
        excitation *= math.exp(-beta * (float(time) - previous))
        log_terms.append(math.log(baseline + alpha * excitation))
        excitation += 1.0
        previous = float(time)
    compensator = baseline * horizon + (alpha * initial_excitation / beta) * (
        1.0 - math.exp(-beta * horizon)
    ) + (alpha / beta) * float(
        np.sum(1.0 - np.exp(-beta * (horizon - times)))
    )
    return float(sum(log_terms) - compensator)


def queue_fill_probability(
    *, queue_ahead: int, own_quantity: int, depletion_intensity: float, horizon: float
) -> float:
    if queue_ahead < 0 or own_quantity <= 0 or depletion_intensity < 0.0 or horizon < 0.0:
        raise ValueError("queue fill inputs are outside their admissible domain")
    required = queue_ahead + own_quantity
    return float(poisson.sf(required - 1, depletion_intensity * horizon))


def joint_order_price_beta(signs: np.ndarray, price_changes: np.ndarray) -> float:
    signs = np.asarray(signs, dtype=float)
    changes = np.asarray(price_changes, dtype=float)
    if signs.ndim != 1 or signs.shape != changes.shape or signs.size < 2:
        raise ValueError("order signs and price changes must be aligned vectors")
    centered = signs - signs.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise ValueError("order signs have no identifying variation")
    return float(centered @ (changes - changes.mean()) / denominator)


@dataclass
class Order:
    order_id: str
    side: str
    price: float
    quantity: int
    sequence: int


class OrderBook:
    def __init__(self) -> None:
        self.orders: list[Order] = []
        self._sequence = 0
        self.traded_quantity = 0

    def add(self, order_id: str, side: str, price: float, quantity: int) -> None:
        if side not in {"buy", "sell"} or not math.isfinite(price) or price <= 0.0 or quantity <= 0:
            raise ValueError("invalid limit order")
        if any(order.order_id == order_id for order in self.orders):
            raise ValueError("duplicate order id")
        opposite = [order for order in self.orders if order.side != side]
        if opposite:
            best = min(order.price for order in opposite) if side == "buy" else max(order.price for order in opposite)
            if (side == "buy" and price >= best) or (side == "sell" and price <= best):
                raise ValueError("crossed limit order requires aggressive execution")
        self._sequence += 1
        self.orders.append(Order(order_id, side, float(price), int(quantity), self._sequence))
        self.assert_invariants()

    def queue_ahead(self, order_id: str) -> int:
        target = next((order for order in self.orders if order.order_id == order_id), None)
        if target is None:
            raise ValueError("unknown live order")
        return sum(
            order.quantity
            for order in self.orders
            if order.side == target.side
            and order.price == target.price
            and order.sequence < target.sequence
        )

    def cancel(self, order_id: str, quantity: int) -> None:
        target = next((order for order in self.orders if order.order_id == order_id), None)
        if target is None or quantity <= 0 or quantity > target.quantity:
            raise ValueError("cancel exceeds live order quantity")
        target.quantity -= quantity
        if target.quantity == 0:
            self.orders.remove(target)
        self.assert_invariants()

    def market(
        self, side: str, quantity: int, *, allow_partial: bool
    ) -> tuple[list[tuple[str, int, float]], int]:
        if side not in {"buy", "sell"} or quantity <= 0:
            raise ValueError("invalid market order")
        passive_side = "sell" if side == "buy" else "buy"
        candidates = [order for order in self.orders if order.side == passive_side]
        candidates.sort(key=lambda order: (order.price if side == "buy" else -order.price, order.sequence))
        available = sum(order.quantity for order in candidates)
        if not allow_partial and available < quantity:
            raise ValueError("insufficient displayed liquidity")
        remaining = quantity
        fills: list[tuple[str, int, float]] = []
        for order in candidates:
            if remaining == 0:
                break
            fill = min(order.quantity, remaining)
            order.quantity -= fill
            remaining -= fill
            self.traded_quantity += fill
            fills.append((order.order_id, fill, order.price))
            if order.quantity == 0:
                self.orders.remove(order)
        self.assert_invariants()
        return fills, remaining

    def best(self, side: str) -> tuple[float, int]:
        levels = [order for order in self.orders if order.side == side]
        if not levels:
            raise ValueError("book side is empty")
        price = max(order.price for order in levels) if side == "buy" else min(order.price for order in levels)
        return price, sum(order.quantity for order in levels if order.price == price)

    def assert_invariants(self) -> None:
        if len({order.order_id for order in self.orders}) != len(self.orders):
            raise AssertionError("order ids are not unique")
        if any(order.quantity <= 0 for order in self.orders):
            raise AssertionError("live order quantity must be positive")
        bids = [order.price for order in self.orders if order.side == "buy"]
        asks = [order.price for order in self.orders if order.side == "sell"]
        if bids and asks and max(bids) >= min(asks):
            raise AssertionError("passive book is crossed")


__all__ = [
    "OrderBook",
    "hawkes_log_likelihood",
    "joint_order_price_beta",
    "poisson_mle",
    "queue_fill_probability",
    "seasonally_adjusted_poisson_mle",
]

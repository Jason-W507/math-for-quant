from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ExecutionResult:
    filled: list[int]
    remaining: int
    implementation_shortfall: float
    stopped: bool


def execution_path(
    *,
    schedule: list[int],
    fill_rates: list[float],
    initial_inventory: int,
    arrival_price: float,
    temporary_impact: float,
    permanent_impact: float,
    stop_after: int,
    maximum_shortfall: float = math.inf,
) -> ExecutionResult:
    if len(schedule) != len(fill_rates) or initial_inventory < 0 or stop_after <= 0:
        raise ValueError("execution schedule, fill rates and stop rule are inconsistent")
    if temporary_impact < 0.0 or permanent_impact < 0.0 or not math.isfinite(arrival_price):
        raise ValueError("execution prices and impact must be finite and nonnegative")
    remaining = initial_inventory
    cumulative = 0
    shortfall = 0.0
    fills: list[int] = []
    stopped = False
    for step, (requested, rate) in enumerate(zip(schedule, fill_rates, strict=True)):
        if step >= stop_after or shortfall >= maximum_shortfall:
            stopped = True
            fills.extend([0] * (len(schedule) - step))
            break
        if requested < 0 or not 0.0 <= rate <= 1.0:
            raise ValueError("requested quantity and fill rate are invalid")
        filled = min(remaining, int(round(requested * rate)))
        price = arrival_price + permanent_impact * cumulative + temporary_impact * filled
        shortfall += filled * (price - arrival_price)
        cumulative += filled
        remaining -= filled
        fills.append(filled)
    return ExecutionResult(fills, remaining, float(shortfall), stopped)


def optimal_execution(
    *,
    inventory: int,
    steps: int,
    temporary_impact: float,
    permanent_impact: float,
    inventory_risk: float,
    maximum_slice: int,
) -> tuple[list[int], float]:
    if inventory < 0 or steps <= 0 or maximum_slice <= 0:
        raise ValueError("inventory, horizon and maximum slice are invalid")
    if temporary_impact <= 0.0 or permanent_impact < 0.0 or inventory_risk < 0.0:
        raise ValueError("impact and inventory risk are invalid")

    @lru_cache(maxsize=None)
    def solve(step: int, remaining: int) -> tuple[float, tuple[int, ...]]:
        if step == steps:
            return (0.0, ()) if remaining == 0 else (math.inf, ())
        best = (math.inf, ())
        executed_before = inventory - remaining
        for quantity in range(min(maximum_slice, remaining) + 1):
            after = remaining - quantity
            future, path = solve(step + 1, after)
            cost = (
                temporary_impact * quantity * quantity
                + permanent_impact * executed_before * quantity
                + inventory_risk * after * after
                + future
            )
            if cost < best[0] - 1e-12:
                best = cost, (quantity,) + path
        return best

    cost, path = solve(0, inventory)
    if not math.isfinite(cost):
        raise ValueError("inventory cannot be completed before the stop horizon")
    return list(path), float(cost)


def inventory_quotes(
    midprice: float, inventory: int, half_spread: float, inventory_skew: float
) -> tuple[float, float]:
    if not math.isfinite(midprice) or midprice <= 0.0 or half_spread <= 0.0 or inventory_skew < 0.0:
        raise ValueError("market-making quote inputs are invalid")
    reservation = midprice - inventory_skew * inventory
    return reservation - half_spread, reservation + half_spread


@dataclass(frozen=True)
class MarketMakingPath:
    bids: list[float]
    asks: list[float]
    inventories: list[int]
    cash: float


def market_making_cycle(
    *,
    midprices: list[float],
    fills: list[str],
    initial_inventory: int,
    half_spread: float,
    inventory_skew: float,
) -> MarketMakingPath:
    if len(midprices) != len(fills):
        raise ValueError("market-making prices and fills must be aligned")
    inventory = initial_inventory
    cash = 0.0
    bids: list[float] = []
    asks: list[float] = []
    inventories = [inventory]
    for midprice, fill in zip(midprices, fills, strict=True):
        bid, ask = inventory_quotes(midprice, inventory, half_spread, inventory_skew)
        bids.append(bid)
        asks.append(ask)
        if fill == "bid":
            inventory += 1
            cash -= bid
        elif fill == "ask":
            inventory -= 1
            cash += ask
        elif fill != "none":
            raise ValueError("unknown market-making fill state")
        inventories.append(inventory)
    return MarketMakingPath(bids, asks, inventories, cash)


__all__ = [
    "ExecutionResult",
    "MarketMakingPath",
    "execution_path",
    "inventory_quotes",
    "market_making_cycle",
    "optimal_execution",
]

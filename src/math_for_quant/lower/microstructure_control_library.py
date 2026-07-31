from __future__ import annotations

import itertools
import math


def enumerate_execution(
    *, inventory: int, steps: int, temporary_impact: float, permanent_impact: float,
    inventory_risk: float, maximum_slice: int,
) -> tuple[list[int], float]:
    if (
        not isinstance(inventory, int)
        or not isinstance(steps, int)
        or not isinstance(maximum_slice, int)
        or inventory < 0
        or steps <= 0
        or maximum_slice <= 0
        or inventory > steps * maximum_slice
        or not all(
            math.isfinite(value)
            for value in (temporary_impact, permanent_impact, inventory_risk)
        )
        or temporary_impact <= 0.0
        or permanent_impact < 0.0
        or inventory_risk < 0.0
    ):
        raise ValueError("invalid enumerated execution inputs")
    best_path: tuple[int, ...] | None = None
    best_cost = math.inf
    for path in itertools.product(range(maximum_slice + 1), repeat=steps):
        if sum(path) != inventory:
            continue
        remaining = inventory
        executed = 0
        cost = 0.0
        for quantity in path:
            remaining -= quantity
            cost += temporary_impact * quantity**2
            cost += permanent_impact * executed * quantity
            cost += inventory_risk * remaining**2
            executed += quantity
        if cost < best_cost - 1e-12:
            best_path, best_cost = path, cost
    if best_path is None:
        raise ValueError("no feasible enumerated schedule")
    return list(best_path), float(best_cost)


def closed_form_inventory_quotes(
    midprice: float, inventory: int, half_spread: float, inventory_skew: float
) -> tuple[float, float]:
    center = midprice - inventory * inventory_skew
    return center - half_spread, center + half_spread


__all__ = ["closed_form_inventory_quotes", "enumerate_execution"]

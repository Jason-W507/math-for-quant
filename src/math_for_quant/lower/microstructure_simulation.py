from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairedSimulationResult:
    events: int
    baseline_signs_hash: str
    control_signs_hash: str
    baseline_fills_hash: str
    control_fills_hash: str
    fills: int
    baseline_pnl: float
    control_pnl: float
    ending_inventory: int


def paired_event_simulation(
    *, seed: int, events: int, base_intensity: float, fill_probability: float = 1.0
) -> PairedSimulationResult:
    """Compare two quoting rules on one common-random-number event stream."""
    if events <= 0 or base_intensity <= 0.0 or not 0.0 <= fill_probability <= 1.0:
        raise ValueError("events, intensity and fill probability are invalid")
    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1.0 / base_intensity, events)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=events)
    fills = rng.random(events) < fill_probability
    innovations = rng.normal(0.0, 0.02, events)
    mid_changes = 0.006 * signs + innovations * np.sqrt(interarrivals)
    digest = hashlib.sha256(signs.tobytes()).hexdigest()
    fills_digest = hashlib.sha256(fills.tobytes()).hexdigest()

    half_spread = 0.01
    inventory_skew = 0.002
    midprice = 100.0
    baseline_inventory = 0
    baseline_cash = 0.0
    control_inventory = 0
    control_cash = 0.0
    for sign, change, filled in zip(signs, mid_changes, fills, strict=True):
        if not filled:
            midprice += float(change)
            continue
        fill_inventory = -int(sign)  # aggressive buy => maker sells one unit
        baseline_quote = midprice + float(sign) * half_spread
        control_reservation = midprice - inventory_skew * control_inventory
        control_quote = control_reservation + float(sign) * half_spread
        baseline_cash -= fill_inventory * baseline_quote
        control_cash -= fill_inventory * control_quote
        baseline_inventory += fill_inventory
        control_inventory += fill_inventory
        midprice += float(change)
    baseline_pnl = baseline_cash + baseline_inventory * midprice
    control_pnl = control_cash + control_inventory * midprice
    return PairedSimulationResult(
        events=events,
        baseline_signs_hash=digest,
        control_signs_hash=digest,
        baseline_fills_hash=fills_digest,
        control_fills_hash=fills_digest,
        fills=int(fills.sum()),
        baseline_pnl=baseline_pnl,
        control_pnl=float(control_pnl),
        ending_inventory=control_inventory,
    )


def analyze_sec_order_placement(path: Path) -> dict[str, float | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("SEC snapshot requires at least two placement categories")
    categories = [str(row.get("category", "")) for row in rows]
    required_categories = {
        "inside_spread", "at_spread", "within_50bp_outside",
        "more_than_50bp_outside", "locked_or_crossed",
    }
    if set(categories) != required_categories or len(categories) != len(required_categories):
        raise ValueError("SEC placement categories must match the declared enumeration")
    shares = np.asarray([float(row["event_share"]) for row in rows], dtype=float)
    ratios = np.asarray([float(row["cancel_to_trade_ratio"]) for row in rows], dtype=float)
    if (
        np.any(~np.isfinite(shares))
        or np.any(~np.isfinite(ratios))
        or np.any(shares < 0.0)
        or np.any(shares > 1.0)
        or np.any(ratios < 0.0)
        or not np.isclose(shares.sum(), 1.0, atol=1e-9)
    ):
        raise ValueError("SEC placement shares and ratios violate their declared domains")
    execution_probabilities = 1.0 / (1.0 + ratios)
    probability_by_category = dict(zip(categories, execution_probabilities, strict=True))
    return {
        "categories": len(rows),
        "event_share_sum": float(shares.sum()),
        "weighted_cancel_to_trade": float(shares @ ratios),
        "implied_execution_probability": float(shares @ execution_probabilities),
        "inside_execution_probability": float(probability_by_category["inside_spread"]),
        "at_spread_execution_probability": float(probability_by_category["at_spread"]),
        "within_50bp_execution_probability": float(
            probability_by_category["within_50bp_outside"]
        ),
        "maximum_cancel_to_trade": float(ratios.max()),
    }


__all__ = ["PairedSimulationResult", "analyze_sec_order_placement", "paired_event_simulation"]

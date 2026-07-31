from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairedSimulationResult:
    events: int
    baseline_signs_hash: str
    control_signs_hash: str
    baseline_random_hash: str
    control_random_hash: str
    baseline_fills: int
    control_fills: int
    baseline_pnl: float
    control_pnl: float
    baseline_ending_inventory: int
    control_ending_inventory: int
    baseline_max_abs_inventory: int
    control_max_abs_inventory: int


def paired_event_simulation(
    *, seed: int, events: int, base_intensity: float, fill_probability: float = 1.0
) -> PairedSimulationResult:
    """Compare two quoting rules on one common-random-number event stream."""
    if (
        not isinstance(events, int)
        or events <= 0
        or not math.isfinite(base_intensity)
        or base_intensity <= 0.0
        or not math.isfinite(fill_probability)
        or not 0.0 <= fill_probability <= 1.0
    ):
        raise ValueError("events, intensity and fill probability are invalid")
    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1.0 / base_intensity, events)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=events)
    fill_uniforms = rng.random(events)
    innovations = rng.normal(0.0, 0.02, events)
    mid_changes = 0.006 * signs + innovations * np.sqrt(interarrivals)
    digest = hashlib.sha256(signs.tobytes()).hexdigest()
    random_digest = hashlib.sha256(fill_uniforms.tobytes()).hexdigest()

    half_spread = 0.01
    inventory_skew = 0.002
    midprice = 100.0
    baseline_inventory = 0
    baseline_cash = 0.0
    control_inventory = 0
    control_cash = 0.0
    baseline_fills = 0
    control_fills = 0
    baseline_max_abs_inventory = 0
    control_max_abs_inventory = 0
    for sign, change, uniform in zip(signs, mid_changes, fill_uniforms, strict=True):
        fill_inventory = -int(sign)  # aggressive buy => maker sells one unit
        baseline_quote = midprice + float(sign) * half_spread
        control_reservation = midprice - inventory_skew * control_inventory
        control_quote = control_reservation + float(sign) * half_spread
        # The same uniform drives both policies, but each quote determines its own
        # fill threshold. A more aggressive quote receives a higher probability.
        quote_advantage = -float(sign) * (control_quote - baseline_quote)
        log_multiplier = float(np.clip(quote_advantage / half_spread, -50.0, 50.0))
        control_probability = float(
            np.clip(fill_probability * math.exp(log_multiplier), 0.0, 1.0)
        )
        if uniform < fill_probability:
            baseline_cash -= fill_inventory * baseline_quote
            baseline_inventory += fill_inventory
            baseline_fills += 1
            baseline_max_abs_inventory = max(
                baseline_max_abs_inventory, abs(baseline_inventory)
            )
        if uniform < control_probability:
            control_cash -= fill_inventory * control_quote
            control_inventory += fill_inventory
            control_fills += 1
            control_max_abs_inventory = max(
                control_max_abs_inventory, abs(control_inventory)
            )
        midprice += float(change)
    baseline_pnl = baseline_cash + baseline_inventory * midprice
    control_pnl = control_cash + control_inventory * midprice
    return PairedSimulationResult(
        events=events,
        baseline_signs_hash=digest,
        control_signs_hash=digest,
        baseline_random_hash=random_digest,
        control_random_hash=random_digest,
        baseline_fills=baseline_fills,
        control_fills=control_fills,
        baseline_pnl=baseline_pnl,
        control_pnl=float(control_pnl),
        baseline_ending_inventory=baseline_inventory,
        control_ending_inventory=control_inventory,
        baseline_max_abs_inventory=baseline_max_abs_inventory,
        control_max_abs_inventory=control_max_abs_inventory,
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

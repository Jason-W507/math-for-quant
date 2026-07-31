from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.microstructure_events import joint_order_price_beta


@dataclass(frozen=True)
class PairedSimulationResult:
    events: int
    baseline_signs_hash: str
    control_signs_hash: str
    baseline_pnl: float
    control_pnl: float
    ending_inventory: int


def paired_event_simulation(
    *, seed: int, events: int, base_intensity: float
) -> PairedSimulationResult:
    """Compare two quoting rules on one common-random-number event stream."""
    if events <= 0 or base_intensity <= 0.0:
        raise ValueError("events and base_intensity must be positive")
    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1.0 / base_intensity, events)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=events)
    innovations = rng.normal(0.0, 0.02, events)
    mid_changes = 0.006 * signs + innovations * np.sqrt(interarrivals)
    digest = hashlib.sha256(signs.tobytes()).hexdigest()

    half_spread = 0.01
    baseline_pnl = float(events * half_spread + np.dot(signs, mid_changes))
    inventory = 0
    control_pnl = 0.0
    for sign, change in zip(signs, mid_changes, strict=True):
        # An aggressive buy lifts our ask (inventory falls), and conversely.
        fill_inventory = -int(sign)
        skew_penalty = 0.002 * abs(inventory)
        control_pnl += half_spread - skew_penalty + fill_inventory * float(change)
        inventory += fill_inventory
    return PairedSimulationResult(
        events=events,
        baseline_signs_hash=digest,
        control_signs_hash=digest,
        baseline_pnl=baseline_pnl,
        control_pnl=float(control_pnl),
        ending_inventory=inventory,
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def analyze_coinbase_trades(path: Path) -> dict[str, float | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["trades"] if isinstance(payload, dict) else payload
    if len(rows) < 2:
        raise ValueError("at least two trades are required")
    trade_ids = np.asarray([int(row["trade_id"]) for row in rows], dtype=np.int64)
    times = [_parse_time(str(row["time"])) for row in rows]
    if np.any(np.diff(trade_ids) <= 0):
        raise ValueError("trade ids must be strictly increasing")
    seconds = np.asarray([(time - times[0]).total_seconds() for time in times])
    if np.any(np.diff(seconds) < 0):
        raise ValueError("trade timestamps must be nondecreasing")
    prices = np.asarray([float(row["price"]) for row in rows])
    sizes = np.asarray([float(row["size"]) for row in rows])
    # Coinbase reports maker side: maker sell means an aggressive buy (+1).
    signs = np.asarray([1.0 if row["side"] == "sell" else -1.0 for row in rows])
    changes = np.diff(prices)
    beta = joint_order_price_beta(signs[1:], changes)
    duration = float(seconds[-1])
    return {
        "trades": len(rows),
        "first_trade_id": int(trade_ids[0]),
        "last_trade_id": int(trade_ids[-1]),
        "duration_seconds": duration,
        "mean_interarrival_seconds": duration / (len(rows) - 1),
        "maker_sell_share": float(np.mean(signs > 0)),
        "total_size": float(sizes.sum()),
        "order_price_beta": beta,
    }


__all__ = ["PairedSimulationResult", "analyze_coinbase_trades", "paired_event_simulation"]

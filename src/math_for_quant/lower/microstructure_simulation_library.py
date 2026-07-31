from __future__ import annotations

import math
import numpy as np


def vectorized_static_pnl(
    signs: np.ndarray,
    price_changes: np.ndarray,
    half_spread: float,
    fills: np.ndarray | None = None,
) -> float:
    signs = np.asarray(signs, dtype=float)
    changes = np.asarray(price_changes, dtype=float)
    if signs.shape != changes.shape:
        raise ValueError("simulation arrays must align")
    filled = np.ones(signs.shape, dtype=bool) if fills is None else np.asarray(fills, dtype=bool)
    if filled.shape != signs.shape:
        raise ValueError("fill mask must align with the event stream")
    inventory_changes = np.where(filled, -signs, 0.0)
    mid_before = 100.0 + np.concatenate(([0.0], np.cumsum(changes[:-1])))
    quotes = mid_before + signs * half_spread
    cash = float(np.sum(-inventory_changes * quotes))
    inventory = float(np.sum(inventory_changes))
    terminal_mid = 100.0 + float(np.sum(changes))
    return cash + inventory * terminal_mid


def library_duration_and_beta(
    seconds: np.ndarray, maker_signs: np.ndarray, prices: np.ndarray
) -> tuple[float, float]:
    seconds = np.asarray(seconds, dtype=float)
    signs = np.asarray(maker_signs, dtype=float)[1:]
    changes = np.diff(np.asarray(prices, dtype=float))
    design = np.column_stack([np.ones(signs.size), signs])
    coefficient, *_ = np.linalg.lstsq(design, changes, rcond=None)
    return float(seconds[-1] - seconds[0]), float(coefficient[1])


def independent_inventory_feedback_ledger(
    signs: np.ndarray,
    price_changes: np.ndarray,
    uniforms: np.ndarray,
    fill_probability: float,
    *,
    half_spread: float = 0.01,
    inventory_skew: float = 0.002,
) -> tuple[float, int, int, int]:
    """Reconstruct the feedback policy with an independent state ledger."""
    directions = np.asarray(signs, dtype=int)
    changes = np.asarray(price_changes, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if directions.shape != changes.shape or directions.shape != draws.shape:
        raise ValueError("independent simulation arrays must align")
    mid = 100.0
    cash = 0.0
    position = 0
    fills = 0
    maximum = 0
    for direction, change, draw in zip(directions, changes, draws, strict=True):
        static_quote = mid + int(direction) * half_spread
        feedback_quote = mid - inventory_skew * position + int(direction) * half_spread
        relative_aggressiveness = -int(direction) * (feedback_quote - static_quote)
        multiplier = math.exp(float(np.clip(relative_aggressiveness / half_spread, -50.0, 50.0)))
        threshold = min(1.0, max(0.0, fill_probability * multiplier))
        if draw < threshold:
            position_change = -int(direction)
            cash -= position_change * feedback_quote
            position += position_change
            fills += 1
            maximum = max(maximum, abs(position))
        mid += float(change)
    return cash + position * mid, fills, position, maximum


__all__ = [
    "independent_inventory_feedback_ledger",
    "library_duration_and_beta",
    "vectorized_static_pnl",
]

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.portfolio_estimation import risk_parity_weights


def run_portfolio_real_data(snapshot_path: Path) -> dict[str, float | int]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = snapshot["rows"]
    periods = [str(row["period"]) for row in rows]
    if periods != sorted(periods) or periods[-1] != snapshot["observed_through"]:
        raise ValueError("real-data time protocol rejected")
    levels = np.array(
        [[float(row["realgdp"]), float(row["realcons"])] for row in rows], dtype=float
    )
    if np.any(levels <= 0.0):
        raise ValueError("log-growth inputs must be positive")
    growth = np.diff(np.log(levels), axis=0)
    covariance = np.cov(growth, rowvar=False, ddof=1)
    weights = risk_parity_weights(covariance)
    return {
        "rows": len(rows),
        "growth_rows": growth.shape[0],
        "covariance_trace": float(np.trace(covariance)),
        "risk_parity_weight_1": float(weights[0]),
        "risk_parity_weight_2": float(weights[1]),
    }


__all__ = ["run_portfolio_real_data"]

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from math_for_quant.lower.portfolio_estimation import risk_parity_weights
from math_for_quant.lower.portfolio_optimization import cvar_optimize
from math_for_quant.lower.portfolio_tail import empirical_tail_risk


def run_portfolio_real_data(snapshot_path: Path) -> dict[str, float | int | str]:
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
    cvar = cvar_optimize(growth, confidence=0.75, maximum_weight=0.8)
    portfolio_losses = -(growth @ np.array([0.5, 0.5]))
    tail_confidence = 0.75
    effective_tail_observations = portfolio_losses.size * (1.0 - tail_confidence)
    try:
        empirical_tail_risk(
            portfolio_losses,
            tail_confidence,
            minimum_tail_observations=20,
            warning_tail_observations=30,
        )
    except ValueError as error:
        if "effective tail observations" not in str(error):
            raise
        tail_status = "reject"
    else:
        tail_status = "pass"
    return {
        "rows": len(rows),
        "growth_rows": growth.shape[0],
        "covariance_trace": float(np.trace(covariance)),
        "risk_parity_weight_1": float(weights[0]),
        "risk_parity_weight_2": float(weights[1]),
        "cvar_confidence": 0.75,
        "cvar_weight_1": float(cvar.weights[0]),
        "cvar_weight_2": float(cvar.weights[1]),
        "cvar_objective": float(cvar.objective),
        "tail_confidence": tail_confidence,
        "tail_effective_observations": float(effective_tail_observations),
        "tail_status": tail_status,
    }


__all__ = ["run_portfolio_real_data"]

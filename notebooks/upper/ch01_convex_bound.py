# %% [markdown]
# # 凸组合界与反例
#
# Jupytext 文本源是权威版本；期望值来自独立手算记录。

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def main(oracle_path: Path = Path("evidence/ch01/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    returns = np.array(oracle["returns"], dtype=np.float64)
    weights = np.array(oracle["weights"], dtype=np.float64)
    counterexample_weights = np.array(
        oracle["counterexample_weights"], dtype=np.float64
    )
    tolerance = float(oracle["absolute_tolerance"])
    if returns.size != weights.size:
        raise SystemExit(
            "assumption gate failed: returns and weights must have equal length"
        )
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(weights)):
        raise SystemExit("assumption gate failed: returns and weights must be finite")
    if returns.size != counterexample_weights.size:
        raise SystemExit(
            "counterexample gate failed: returns and weights must have equal length"
        )
    if not np.all(np.isfinite(counterexample_weights)):
        raise SystemExit("counterexample gate failed: weights must be finite")
    if abs(float(np.sum(weights)) - 1.0) > tolerance:
        raise SystemExit("assumption gate failed: weights must sum to one")
    if np.any(weights < 0.0):
        raise SystemExit("assumption gate failed: weights must be nonnegative")
    if abs(float(np.sum(counterexample_weights)) - 1.0) > tolerance:
        raise SystemExit("counterexample gate failed: weights must sum to one")
    if not np.any(counterexample_weights < 0.0):
        raise SystemExit(
            "counterexample gate failed: deleted nonnegativity assumption is absent"
        )

    weighted_return = float(weights @ returns)
    lower = float(np.min(returns))
    upper = float(np.max(returns))
    counterexample = float(counterexample_weights @ returns)
    observed = (weighted_return, lower, upper, counterexample)
    expected = (
        float(oracle["expected_weighted_return"]),
        float(oracle["expected_lower"]),
        float(oracle["expected_upper"]),
        float(oracle["expected_counterexample"]),
    )
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")
    if not lower <= weighted_return <= upper:
        raise SystemExit("convex bound unexpectedly failed")
    if counterexample <= upper:
        raise SystemExit("deleted-assumption counterexample was not observed")

    print(
        "oracle=passed "
        f"weighted_return={weighted_return:.6f} lower={lower:.6f} "
        f"upper={upper:.6f} counterexample={counterexample:.6f}"
    )
    return 0


oracle_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "evidence/ch01/oracle.json"
)
main(oracle_path)

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def independent_log_level_slope(rows: list[dict[str, object]]) -> float:
    """Compute OLS with scalar arithmetic, independent of route implementations."""
    x = [math.log(float(row["realcons"])) for row in rows]
    y = [math.log(float(row["realgdp"])) for row in rows]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        raise SystemExit("real-data oracle has zero regressor variance")
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    snapshot_path = Path(oracle["fixture"]["path"])
    observed_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if observed_hash != oracle["fixture"]["sha256"]:
        raise SystemExit("real-data snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = snapshot["rows"]
    periods = [row["period"] for row in rows]
    if periods != sorted(periods) or periods[-1] != snapshot["observed_through"]:
        raise SystemExit("real-data time protocol rejected")
    observed = {
        "rows": len(rows),
        "slope": independent_log_level_slope(rows),
    }
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(
                f"real-data {key} mismatch: observed={observed[key]} expected={expected}"
            )
    print(
        "real-data-oracle=passed "
        f"rows={observed['rows']} slope={observed['slope']:.6f}"
    )
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "evidence/stat-arb/real-data-oracle.json"
    )
    raise SystemExit(main(path))

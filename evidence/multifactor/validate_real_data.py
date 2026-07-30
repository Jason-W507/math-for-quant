from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    return covariance / math.sqrt(left_ss * right_ss)


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    snapshot_path = Path(oracle["fixture"]["path"])
    observed_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if observed_hash != oracle["fixture"]["sha256"]:
        raise SystemExit("real-data snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if int(snapshot["signal_year"]) >= int(snapshot["outcome_year"]):
        raise SystemExit("real-data time protocol rejected")
    rows = snapshot["rows"]
    observed = {
        "rows": len(rows),
        "signal_year": int(snapshot["signal_year"]),
        "outcome_year": int(snapshot["outcome_year"]),
        "correlation": correlation(
            [float(row["signal"]) for row in rows],
            [float(row["outcome"]) for row in rows],
        ),
    }
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(
                f"real-data {key} mismatch: observed={observed[key]} expected={expected}"
            )
    print(
        "real-data-oracle=passed "
        f"rows={observed['rows']} years=({observed['signal_year']},{observed['outcome_year']}) "
        f"correlation={observed['correlation']:.6f}"
    )
    return 0


if __name__ == "__main__":
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("evidence/multifactor/real-data-oracle.json")
    )
    raise SystemExit(main(path))

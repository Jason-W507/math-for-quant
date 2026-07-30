from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from math_for_quant.lower.stat_arb_library import cross_check_long_run


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
    y = np.log(np.asarray([row["realgdp"] for row in rows], dtype=float))
    x = np.log(np.asarray([row["realcons"] for row in rows], dtype=float))
    check = cross_check_long_run(y, x)
    observed = {
        "rows": len(rows),
        "slope": check.transparent_slope,
        "slope_gap": abs(check.transparent_slope - check.library_slope),
    }
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(
                f"real-data {key} mismatch: observed={observed[key]} expected={expected}"
            )
    print(
        "real-data-oracle=passed "
        f"rows={observed['rows']} slope={observed['slope']:.6f} "
        f"slope_gap={observed['slope_gap']:.3e}"
    )
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "evidence/stat-arb/real-data-oracle.json"
    )
    raise SystemExit(main(path))

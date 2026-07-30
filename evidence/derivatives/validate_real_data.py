from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    snapshot_path = Path(oracle["snapshot"])
    raw = snapshot_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != oracle["sha256"]:
        raise SystemExit("Treasury snapshot hash drifted")
    snapshot = json.loads(raw)
    if snapshot["observation_date"] >= oracle["decision_date"]:
        raise SystemExit("Treasury observation must predate the decision")
    maturities = {"3_month": 0.25, "6_month": 0.5, "1_year": 1.0}
    for name, maturity in maturities.items():
        observed = math.exp(-snapshot["rates_percent"][name] / 100.0 * maturity)
        expected = oracle["expected_discount_factors"][name]
        if abs(observed - expected) > oracle["absolute_tolerance"]:
            raise SystemExit(f"discount factor drifted: {name}")
    print("derivatives-real-data=passed observations=3 decision=2024-12-03")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives/real-data-oracle.json")))

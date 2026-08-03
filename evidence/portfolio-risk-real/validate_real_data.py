from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from m4q.lower.portfolio_real_data import run_portfolio_real_data


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    snapshot_path = Path(oracle["fixture"]["path"])
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != oracle["fixture"]["sha256"]:
        raise SystemExit("portfolio real-data snapshot hash mismatch")
    observed = run_portfolio_real_data(snapshot_path)
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if isinstance(expected, str):
            if observed[key] != expected:
                raise SystemExit(
                    f"portfolio real-data {key} mismatch: observed={observed[key]} expected={expected}"
                )
            continue
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(
                f"portfolio real-data {key} mismatch: observed={observed[key]} expected={expected}"
            )
    print(
        "portfolio-real-data=passed "
        + " ".join(
            f"{key}={value}" if isinstance(value, str) else f"{key}={value:.8g}"
            for key, value in observed.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            Path(sys.argv[1])
            if len(sys.argv) > 1
            else Path("evidence/portfolio-risk-real/real-data-oracle.json")
        )
    )

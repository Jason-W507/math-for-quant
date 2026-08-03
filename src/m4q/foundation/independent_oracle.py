from __future__ import annotations

from pathlib import Path

import numpy as np

from m4q.evidence import load_oracle_bundle


def main(oracle_path: Path = Path("evidence/foundation/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    returns = np.array(oracle["returns"], dtype=np.float64)
    observed = float(np.mean(returns))
    expected = float(oracle["expected"])
    tolerance = float(oracle["absolute_tolerance"])
    if abs(observed - expected) > tolerance:
        raise SystemExit(
            f"oracle mismatch: observed={observed:.12f} expected={expected:.12f}"
        )
    print(f"oracle=passed observed={observed:.12f} expected={expected:.12f}")
    return 0


if __name__ == "__main__":
    main()

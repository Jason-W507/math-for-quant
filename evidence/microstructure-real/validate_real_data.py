from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from math_for_quant.lower.microstructure_simulation import analyze_coinbase_trades


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    snapshot = Path(oracle["snapshot"]["path"])
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != oracle["snapshot"]["sha256"]:
        raise SystemExit("Coinbase snapshot hash mismatch")
    observed = analyze_coinbase_trades(snapshot)
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(f"{key} mismatch")
    print("microstructure-real-data=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-real/real-data-oracle.json")))

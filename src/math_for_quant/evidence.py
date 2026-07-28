from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_oracle_bundle(oracle_path: Path) -> dict[str, object]:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    fixture_contract = oracle.get("fixture")
    if not isinstance(fixture_contract, dict):
        raise SystemExit("fixture gate failed: oracle must declare a fixture")
    fixture_path = ROOT / str(fixture_contract.get("path", ""))
    fixture_bytes = fixture_path.read_bytes()
    observed_digest = hashlib.sha256(fixture_bytes).hexdigest()
    expected_digest = str(fixture_contract.get("sha256", ""))
    if observed_digest != expected_digest:
        raise SystemExit(
            "fixture hash mismatch: "
            f"observed={observed_digest} expected={expected_digest}"
        )
    fixture = json.loads(fixture_bytes)
    if not isinstance(fixture, dict):
        raise SystemExit("fixture gate failed: fixture root must be an object")
    overlap = set(fixture).intersection(oracle).difference({"fixture"})
    if overlap:
        names = ", ".join(sorted(overlap))
        raise SystemExit(f"fixture gate failed: duplicate fields {names}")
    return {**fixture, **oracle}

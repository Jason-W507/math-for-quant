from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs" / "template-provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the external ElegantBook baseline was not modified."
    )
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    args = parse_args()
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    files = provenance["files"]
    for record in files:
        name = record["source"]
        expected = record["sha256"]
        path = args.source / name
        if not path.is_file():
            print(f"missing external template file: {path}", file=sys.stderr)
            return 1
        observed = sha256(path)
        if observed != expected:
            print(
                f"external template changed: {name} "
                f"expected={expected} observed={observed}",
                file=sys.stderr,
            )
            return 1

    vendored = [record for record in files if record.get("vendored")]
    for record in vendored:
        path = ROOT / record["vendored"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            print(
                f"vendored asset does not match its source: {record['vendored']}",
                file=sys.stderr,
            )
            return 1

    print(f"template={provenance['template']}")
    print(f"external-baseline=passed files={len(files)}")
    print(f"vendored-assets=passed files={len(vendored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

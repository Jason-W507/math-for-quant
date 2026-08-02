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
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--vendored-only",
        action="store_true",
        help="Verify committed vendored assets when the external baseline is unavailable.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    args = parse_args()
    if args.source is None and not args.vendored_only:
        print("provide --source or --vendored-only", file=sys.stderr)
        return 2
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    files = provenance["files"]
    if args.source is not None:
        declared = {str(record["source"]) for record in files}
        observed = {
            path.relative_to(args.source).as_posix()
            for path in args.source.rglob("*")
            if path.is_file()
        }
        if declared != observed:
            missing = sorted(declared - observed)
            extra = sorted(observed - declared)
            print(
                f"external template inventory changed: missing={missing} extra={extra}",
                file=sys.stderr,
            )
            return 1
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
        expected = record.get("vendored_sha256", record["sha256"])
        if not path.is_file() or sha256(path) != expected:
            print(
                f"vendored asset does not match its source: {record['vendored']}",
                file=sys.stderr,
            )
            return 1

    print(f"template={provenance['template']}")
    if args.source is not None:
        print(f"external-baseline=passed files={len(files)}")
    else:
        print("external-baseline=not-available ci-vendored-check=used")
    print(f"vendored-assets=passed files={len(vendored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

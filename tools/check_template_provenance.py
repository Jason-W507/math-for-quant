from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "elegantbook.cls": "D2CDB802B3DE46B1D659D1A8EB36979AECD761402D3E95296936F433F97549EB",
    "License": "5F05FCF6EF25A6C31BCCD2DF7C0C46B23107BBEB2CE5CDBA74EFB5CC357F4DBB",
    "figure/cover.jpg": "0748354F5D61633F9032DAB0A4A6774CB91CF1A0FC5892CC52D73A47A4552A0B",
}


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
    for name, expected in EXPECTED.items():
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

    vendored_license = ROOT / "vendor" / "elegantbook" / "LPPL-License.txt"
    if not vendored_license.is_file() or sha256(vendored_license) != EXPECTED["License"]:
        print("vendored LPPL license does not match its source", file=sys.stderr)
        return 1
    vendored_cover = ROOT / "assets" / "cover.jpg"
    if not vendored_cover.is_file() or sha256(vendored_cover) != EXPECTED["figure/cover.jpg"]:
        print("vendored cover does not match its source", file=sys.stderr)
        return 1

    print("template=ElegantBook-4.7")
    print("external-baseline=passed files=3")
    print("vendored-assets=passed files=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

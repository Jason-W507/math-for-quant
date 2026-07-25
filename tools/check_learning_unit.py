from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contract.curriculum import load_registries, validate_prerequisites, validate_shared
from contract.evidence import (
    REQUIRED_EVIDENCE_FIELDS,
    artifact_paths,
    run_oracle,
    validate_content,
)
from contract.publication import validate as validate_publications


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate public learning-unit evidence packages."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit")
    parser.add_argument("--volume", choices=("upper", "lower", "all"))
    return parser.parse_args()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    if args.unit is not None and args.volume is not None:
        return fail("choose either --unit or --volume")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    units = manifest.get("units", [])

    prerequisite_error = validate_prerequisites(units)
    if prerequisite_error is not None:
        return fail(prerequisite_error)

    selected = None
    accepted: list[dict[str, object]] = []
    for unit in units:
        if args.unit is not None and unit.get("id") == args.unit:
            selected = unit
        if unit.get("state") != "accepted":
            continue
        accepted.append(unit)
        evidence = unit.get("evidence", {})
        for field in REQUIRED_EVIDENCE_FIELDS:
            if field not in evidence:
                return fail(
                    f"{unit.get('id', '<unknown>')}: missing evidence field {field}"
                )
        if unit.get("published"):
            for field in (
                "notation_and_assumptions",
                "core_derivation",
                "questions",
                "hints",
                "solutions",
                "capstone_connection",
            ):
                if not str(evidence[field]).startswith("tex/"):
                    return fail(
                        f"{unit.get('id')}: published evidence field {field} "
                        "must reference tex source"
                    )
        try:
            paths = artifact_paths(evidence)
        except (KeyError, TypeError, ValueError) as error:
            return fail(f"{unit.get('id')}: invalid evidence: {error}")
        for relative in paths:
            if not (ROOT / relative).is_file():
                return fail(f"{unit.get('id')}: missing artifact {relative}")

    registry_error, symbols, terms = load_registries(manifest, units, ROOT)
    if registry_error is not None:
        return fail(registry_error)
    for unit in accepted:
        content_error = validate_content(
            str(unit.get("id")), unit["evidence"], ROOT, symbols, terms
        )
        if content_error is not None:
            return fail(content_error)

    if args.unit is not None:
        if selected is None:
            return fail(f"unknown unit: {args.unit}")
        if selected.get("state") != "accepted":
            return fail(f"{args.unit}: unit is not accepted")
        result = run_oracle(selected["evidence"], ROOT)
        if result.returncode != 0:
            return fail(f"{args.unit}: oracle failed: {result.stderr.strip()}")
        print(f"unit={args.unit}")
        print(
            f"evidence={len(REQUIRED_EVIDENCE_FIELDS)}/"
            f"{len(REQUIRED_EVIDENCE_FIELDS)}"
        )
        print(result.stdout, end="")
        print("learning-unit contract passed")
        return 0

    shared_error = validate_shared(manifest, units, ROOT)
    if shared_error is not None:
        return fail(shared_error)
    for unit in accepted:
        result = run_oracle(unit["evidence"], ROOT)
        if result.returncode != 0:
            return fail(f"{unit.get('id')}: oracle failed: {result.stderr.strip()}")

    publication_results: list[tuple[str, int]] = []
    if args.volume is not None:
        publication_error, publication_results = validate_publications(
            manifest, args.volume, ROOT
        )
        if publication_error is not None:
            return fail(publication_error)

    print("curriculum=passed volumes=2 upper_chapters=17 tracks=6")
    print("question-levels=passed count=4")
    print("registries=passed count=2")
    print(f"course-graph=passed units={len(units)}")
    print(f"accepted-units={len(accepted)}")
    for identifier, pages in publication_results:
        print(f"publication=passed volume={identifier} pages={pages}")
    print("learning-unit contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

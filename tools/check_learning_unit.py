from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    from tools.contract.curriculum import (
        load_registries,
        validate_prerequisites,
        validate_shared,
    )
    from tools.contract.evidence import (
        REQUIRED_EVIDENCE_FIELDS,
        artifact_paths,
        run_oracle,
        validate_content,
    )
    from tools.contract.publication import validate as validate_publications
    from tools.contract.schema import validate_document
except ModuleNotFoundError:  # Direct execution puts tools/ rather than the root on sys.path.
    from contract.curriculum import load_registries, validate_prerequisites, validate_shared
    from contract.evidence import (
        REQUIRED_EVIDENCE_FIELDS,
        artifact_paths,
        run_oracle,
        validate_content,
    )
    from contract.publication import validate as validate_publications
    from contract.schema import validate_document


ROOT = Path(__file__).resolve().parents[1]


def validate_route_evidence(unit: dict[str, object]) -> str | None:
    evidence = unit["evidence"]
    data = evidence["dual_track_data"]
    implementation = evidence["dual_implementation"]
    report = evidence["route_report"]
    paths = [
        data["oracle_fixture"],
        data["real_data_snapshot"],
        data["license"],
        data["time_protocol"],
        implementation["transparent"],
        implementation["library"],
        evidence["teaching_notebook"],
        report["source"],
        report["expected"],
        evidence["shared_solutions"],
    ]
    for relative in paths:
        if not (ROOT / relative).is_file():
            return f"{unit['id']}: missing route artifact {relative}"
    snapshot = ROOT / data["real_data_snapshot"]
    observed_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if observed_hash != data["sha256"]:
        return f"{unit['id']}: real-data snapshot hash differs from manifest"
    command = [sys.executable if part == "{python}" else part for part in report["command"]]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return f"{unit['id']}: route report failed: {result.stderr.strip()}"
    expected = (ROOT / report["expected"]).read_text(encoding="utf-8")
    if result.stdout != expected:
        return f"{unit['id']}: route report differs from registered expected output"
    return None


def selected_track_units(
    track: dict[str, object], accepted: list[dict[str, object]]
) -> tuple[list[dict[str, object]], str | None]:
    identifier = str(track["id"])
    units = [
        unit
        for unit in accepted
        if unit.get("track") == identifier and unit.get("published")
    ]
    required_stages = {
        "model-math",
        "estimation-numerics",
        "oos-frictions-capstone",
    }
    observed_stages = {str(unit.get("track_stage")) for unit in units}
    if len(units) != 3 or observed_stages != required_stages:
        return units, (
            f"{identifier}: route requires exactly three accepted learning units "
            "covering model-math, estimation-numerics, and oos-frictions-capstone"
        )
    planned = set(track.get("planned_units", []))
    observed = {str(unit.get("id")) for unit in units}
    if observed != planned:
        return units, f"{identifier}: accepted learning units differ from the route plan"
    return units, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate public learning-unit evidence packages."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit")
    parser.add_argument("--track")
    parser.add_argument("--volume", choices=("upper", "lower", "all"))
    return parser.parse_args()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    selections = sum(value is not None for value in (args.unit, args.track, args.volume))
    if selections > 1:
        return fail("choose exactly one of --unit, --track, or --volume")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return fail(f"manifest JSON is invalid: {error.msg}")
    schema_error = validate_document(manifest, "manifest")
    if not isinstance(manifest, dict):
        return fail(schema_error or "manifest schema validation failed: root must be an object")
    units = manifest.get("units", [])
    if not isinstance(units, list):
        return fail(schema_error or "manifest schema validation failed: units must be an array")

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

    if schema_error is not None:
        return fail(schema_error)

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

    if args.track is not None:
        track = next(
            (item for item in manifest.get("tracks", []) if item.get("id") == args.track),
            None,
        )
        if track is None:
            return fail(f"unknown track: {args.track}")
        track_units, track_error = selected_track_units(track, accepted)
        if track_error is not None:
            return fail(track_error)
        for unit in track_units:
            route_error = validate_route_evidence(unit)
            if route_error is not None:
                return fail(route_error)
            result = run_oracle(unit["evidence"], ROOT)
            if result.returncode != 0:
                return fail(f"{unit.get('id')}: oracle failed: {result.stderr.strip()}")
        print(f"track={args.track} learning-units=3")
        print("track contract passed")
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

    volumes = manifest["volumes"]
    upper_chapters = sum(
        unit.get("volume") == "upper" and not unit.get("internal", False)
        for unit in units
    )
    print(
        f"curriculum=passed volumes={len(volumes)} "
        f"upper_chapters={upper_chapters} tracks={len(manifest['tracks'])}"
    )
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

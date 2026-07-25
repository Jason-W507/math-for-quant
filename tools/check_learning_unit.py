from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVIDENCE_FIELDS = (
    "notation_and_assumptions",
    "core_derivation",
    "independent_oracle",
    "questions",
    "hints",
    "solutions",
    "capstone_connection",
)


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


def artifact_paths(evidence: dict[str, object]) -> list[str]:
    paths: list[str] = []
    for field in REQUIRED_EVIDENCE_FIELDS:
        value = evidence[field]
        if field == "independent_oracle":
            oracle = value
            if not isinstance(oracle, dict):
                raise ValueError("independent_oracle must be an object")
            paths.extend([str(oracle["source"]), str(oracle["oracle"])])
        else:
            paths.append(str(value))
    return paths


def validate_evidence_content(
    identifier: str, evidence: dict[str, object]
) -> str | None:
    for relative in artifact_paths(evidence):
        path = ROOT / relative
        if path.stat().st_size == 0:
            return f"{identifier}: empty evidence artifact {relative}"

    markdown: dict[str, str] = {}
    for field in (
        "notation_and_assumptions",
        "core_derivation",
        "questions",
        "hints",
        "solutions",
        "capstone_connection",
    ):
        markdown[field] = (ROOT / str(evidence[field])).read_text(encoding="utf-8")

    required_question_labels = ("口述概念", "笔试推导", "数值编程", "研究判断")
    if any(label not in markdown["questions"] for label in required_question_labels):
        return (
            f"{identifier}: questions must include "
            + ", ".join(required_question_labels)
        )
    for field in ("hints", "solutions"):
        numbered = re.findall(r"(?m)^\s*[1-4][.)、]", markdown[field])
        if len(numbered) < 4:
            return f"{identifier}: {field} must address all four question levels"
    if not markdown["notation_and_assumptions"].lstrip().startswith("#"):
        return f"{identifier}: notation and assumptions must have a heading"
    if "=" not in markdown["core_derivation"] or "#" not in markdown["core_derivation"]:
        return f"{identifier}: core derivation must contain a heading and derivation"
    if "capstone" not in markdown["capstone_connection"].casefold():
        return f"{identifier}: capstone connection must name the Capstone boundary"

    oracle = evidence["independent_oracle"]
    try:
        oracle_data = json.loads((ROOT / str(oracle["oracle"])).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return f"{identifier}: invalid oracle evidence: {error}"
    for field in ("expected", "absolute_tolerance", "provenance"):
        if field not in oracle_data:
            return f"{identifier}: oracle evidence missing {field}"
    return None


def validate_prerequisites(units: list[dict[str, object]]) -> str | None:
    identifiers = {str(unit.get("id")) for unit in units}
    prerequisites_by_id: dict[str, list[str]] = {}
    for unit in units:
        identifier = str(unit.get("id", "<unknown>"))
        prerequisites = [str(value) for value in unit.get("prerequisites", [])]
        prerequisites_by_id[identifier] = prerequisites
        for prerequisite in prerequisites:
            if prerequisite not in identifiers:
                return f"{identifier}: unknown prerequisite {prerequisite}"

    visited: set[str] = set()
    active: set[str] = set()

    def visit(identifier: str) -> str | None:
        if identifier in active:
            return identifier
        if identifier in visited:
            return None
        active.add(identifier)
        for prerequisite in prerequisites_by_id[identifier]:
            cycle_at = visit(prerequisite)
            if cycle_at is not None:
                return cycle_at
        active.remove(identifier)
        visited.add(identifier)
        return None

    for identifier in prerequisites_by_id:
        cycle_at = visit(identifier)
        if cycle_at is not None:
            return f"course graph contains a prerequisite cycle at {cycle_at}"
    return None


def validate_shared_contract(
    manifest: dict[str, object], units: list[dict[str, object]]
) -> str | None:
    if manifest.get("schema_version") != 1:
        return "unsupported curriculum schema_version"
    if manifest.get("question_levels") != [
        "oral",
        "derivation",
        "computation",
        "research",
    ]:
        return "question levels must be oral, derivation, computation, research"

    volumes = manifest.get("volumes")
    if not isinstance(volumes, list) or {item.get("id") for item in volumes} != {
        "upper",
        "lower",
    }:
        return "curriculum must define upper and lower volumes"
    for volume in volumes:
        source = ROOT / str(volume.get("source", ""))
        if not source.is_file():
            return f"missing volume source: {volume.get('source')}"

    identifiers = [str(unit.get("id")) for unit in units]
    if len(identifiers) != len(set(identifiers)):
        return "course graph contains duplicate unit identifiers"
    expected_upper = {f"upper.ch{number:02d}" for number in range(1, 18)}
    upper = {identifier for identifier in identifiers if identifier.startswith("upper.ch")}
    if upper != expected_upper:
        return "curriculum must define upper.ch01 through upper.ch17"

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 6:
        return "curriculum must define six direction tracks"
    track_ids = [str(track.get("id")) for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        return "curriculum contains duplicate track identifiers"
    unit_ids = set(identifiers)
    for track in tracks:
        for prerequisite in track.get("bridge_prerequisites", []):
            if prerequisite not in unit_ids:
                return f"{track.get('id')}: unknown bridge prerequisite {prerequisite}"

    registries = manifest.get("registries")
    if not isinstance(registries, dict) or set(registries) != {"notation", "glossary"}:
        return "curriculum must define notation and glossary registries"
    for name, relative in registries.items():
        path = ROOT / str(relative)
        if not path.is_file():
            return f"missing {name} registry: {relative}"
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return f"invalid {name} registry JSON: {error.msg}"
        if registry.get("schema_version") != 1:
            return f"unsupported {name} registry schema_version"
        if name == "notation":
            symbols = registry.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                return "notation registry must contain symbols"
            for symbol in symbols:
                if not isinstance(symbol, dict) or any(
                    not str(symbol.get(field, "")).strip()
                    for field in ("symbol", "meaning", "domain", "first_unit")
                ):
                    return "notation registry contains an incomplete symbol"
                if symbol["first_unit"] not in unit_ids:
                    return (
                        "notation registry contains unknown first_unit "
                        f"{symbol['first_unit']}"
                    )
        if name == "glossary":
            terms = registry.get("terms")
            if not isinstance(terms, list) or not terms:
                return "glossary registry must contain terms"
            for term in terms:
                if not isinstance(term, dict) or any(
                    not str(term.get(field, "")).strip()
                    for field in ("zh", "en", "definition")
                ):
                    return "glossary registry contains an incomplete term"
    return None


def run_oracle(evidence: dict[str, object]) -> subprocess.CompletedProcess[str]:
    oracle = evidence["independent_oracle"]
    command = [
        sys.executable if part == "{python}" else str(part)
        for part in oracle["command"]
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def validate_publications(
    manifest: dict[str, object], requested: str
) -> tuple[str | None, list[tuple[str, int]]]:
    volumes = manifest.get("volumes", [])
    selected = [
        volume
        for volume in volumes
        if requested == "all" or volume.get("id") == requested
    ]
    if not selected:
        return f"unknown volume: {requested}", []

    results: list[tuple[str, int]] = []
    for volume in selected:
        identifier = str(volume.get("id", "<unknown>"))
        relative = str(volume.get("pdf", ""))
        path = ROOT / relative
        if not path.is_file():
            return f"{identifier}: missing publication artifact {relative}", []
        try:
            pages = len(PdfReader(path).pages)
        except Exception as error:  # pypdf exposes several parser-specific errors.
            return f"{identifier}: invalid publication artifact {relative}: {error}", []
        if pages < 1:
            return f"{identifier}: publication artifact has no pages {relative}", []
        results.append((identifier, pages))
    return None, results


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
                    f"{unit.get('id', '<unknown>')}: "
                    f"missing evidence field {field}"
                )

        try:
            paths = artifact_paths(evidence)
        except (KeyError, TypeError, ValueError) as error:
            return fail(f"{unit.get('id')}: invalid evidence: {error}")
        for relative in paths:
            if not (ROOT / relative).is_file():
                return fail(f"{unit.get('id')}: missing artifact {relative}")
        content_error = validate_evidence_content(str(unit.get("id")), evidence)
        if content_error is not None:
            return fail(content_error)

    if args.unit is None:
        shared_error = validate_shared_contract(manifest, units)
        if shared_error is not None:
            return fail(shared_error)
        for unit in accepted:
            result = run_oracle(unit["evidence"])
            if result.returncode != 0:
                return fail(
                    f"{unit.get('id')}: oracle failed: {result.stderr.strip()}"
                )
        publication_results: list[tuple[str, int]] = []
        if args.volume is not None:
            publication_error, publication_results = validate_publications(
                manifest, args.volume
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
    if selected is None:
        return fail(f"unknown unit: {args.unit}")
    if selected.get("state") != "accepted":
        return fail(f"{args.unit}: unit is not accepted")

    evidence = selected["evidence"]
    result = run_oracle(evidence)
    if result.returncode != 0:
        return fail(f"{args.unit}: oracle failed: {result.stderr.strip()}")

    print(f"unit={args.unit}")
    print(f"evidence={len(REQUIRED_EVIDENCE_FIELDS)}/{len(REQUIRED_EVIDENCE_FIELDS)}")
    print(result.stdout, end="")

    print("learning-unit contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

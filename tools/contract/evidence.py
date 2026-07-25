from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_EVIDENCE_FIELDS = (
    "notation_and_assumptions",
    "core_derivation",
    "independent_oracle",
    "questions",
    "hints",
    "solutions",
    "capstone_connection",
)


def artifact_paths(evidence: dict[str, object]) -> list[str]:
    paths: list[str] = []
    for field in REQUIRED_EVIDENCE_FIELDS:
        value = evidence[field]
        if field == "independent_oracle":
            if not isinstance(value, dict):
                raise ValueError("independent_oracle must be an object")
            paths.extend([str(value["source"]), str(value["oracle"])])
        else:
            paths.append(str(value))
    return paths


def validate_content(
    identifier: str,
    evidence: dict[str, object],
    root: Path,
    registered_symbols: set[str],
    registered_terms: set[str],
) -> str | None:
    for relative in artifact_paths(evidence):
        if (root / relative).stat().st_size == 0:
            return f"{identifier}: empty evidence artifact {relative}"

    declared_symbols = evidence.get("notation_symbols")
    if not isinstance(declared_symbols, list) or not declared_symbols:
        return f"{identifier}: evidence must declare notation_symbols"
    for symbol in declared_symbols:
        if symbol not in registered_symbols:
            return f"{identifier}: unregistered notation symbol {symbol}"

    declared_terms = evidence.get("glossary_terms")
    if not isinstance(declared_terms, list) or not declared_terms:
        return f"{identifier}: evidence must declare glossary_terms"
    for term in declared_terms:
        if term not in registered_terms:
            return f"{identifier}: unregistered glossary term {term}"

    markdown = {
        field: (root / str(evidence[field])).read_text(encoding="utf-8")
        for field in (
            "notation_and_assumptions",
            "core_derivation",
            "questions",
            "hints",
            "solutions",
            "capstone_connection",
        )
    }
    for symbol in declared_symbols:
        if symbol not in markdown["notation_and_assumptions"]:
            return f"{identifier}: declared notation symbol absent from evidence {symbol}"

    question_labels = ("口述概念", "笔试推导", "数值编程", "研究判断")
    if any(label not in markdown["questions"] for label in question_labels):
        return f"{identifier}: questions must include " + ", ".join(question_labels)
    for field in ("hints", "solutions"):
        if len(re.findall(r"(?m)^\s*[1-4][.)、]", markdown[field])) < 4:
            return f"{identifier}: {field} must address all four question levels"
    if not markdown["notation_and_assumptions"].lstrip().startswith("#"):
        return f"{identifier}: notation and assumptions must have a heading"
    if "=" not in markdown["core_derivation"] or "#" not in markdown["core_derivation"]:
        return f"{identifier}: core derivation must contain a heading and derivation"
    if "capstone" not in markdown["capstone_connection"].casefold():
        return f"{identifier}: capstone connection must name the Capstone boundary"

    oracle = evidence["independent_oracle"]
    try:
        oracle_data = json.loads(
            (root / str(oracle["oracle"])).read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return f"{identifier}: invalid oracle evidence: {error}"
    for field in ("expected", "absolute_tolerance", "provenance"):
        if field not in oracle_data:
            return f"{identifier}: oracle evidence missing {field}"
    return None


def run_oracle(
    evidence: dict[str, object], root: Path
) -> subprocess.CompletedProcess[str]:
    oracle = evidence["independent_oracle"]
    command = [
        sys.executable if part == "{python}" else str(part)
        for part in oracle["command"]
    ]
    return subprocess.run(
        command, cwd=root, text=True, capture_output=True, check=False
    )

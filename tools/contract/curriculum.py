from __future__ import annotations

import json
from pathlib import Path

from .schema import validate_document


def validate_prerequisites(units: list[dict[str, object]]) -> str | None:
    identifiers = {str(unit.get("id")) for unit in units}
    graph: dict[str, list[str]] = {}
    for unit in units:
        identifier = str(unit.get("id", "<unknown>"))
        graph[identifier] = [str(value) for value in unit.get("prerequisites", [])]
        for prerequisite in graph[identifier]:
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
        for prerequisite in graph[identifier]:
            cycle_at = visit(prerequisite)
            if cycle_at is not None:
                return cycle_at
        active.remove(identifier)
        visited.add(identifier)
        return None

    for identifier in graph:
        cycle_at = visit(identifier)
        if cycle_at is not None:
            return f"course graph contains a prerequisite cycle at {cycle_at}"
    return None


def load_registries(
    manifest: dict[str, object], units: list[dict[str, object]], root: Path
) -> tuple[str | None, set[str], set[str]]:
    registries = manifest.get("registries")
    if not isinstance(registries, dict) or set(registries) != {"notation", "glossary"}:
        return "curriculum must define notation and glossary registries", set(), set()
    loaded: dict[str, dict[str, object]] = {}
    for name, relative in registries.items():
        path = root / str(relative)
        if not path.is_file():
            return f"missing {name} registry: {relative}", set(), set()
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return f"invalid {name} registry JSON: {error.msg}", set(), set()
        if loaded[name].get("schema_version") != 1:
            return f"unsupported {name} registry schema_version", set(), set()
        collection_name = "symbols" if name == "notation" else "terms"
        collection = loaded[name].get(collection_name)
        if not isinstance(collection, list) or not collection:
            return f"{name} registry must contain {collection_name}", set(), set()
        schema_error = validate_document(loaded[name], name)
        if schema_error is not None:
            return schema_error, set(), set()

    unit_ids = {str(unit.get("id")) for unit in units}
    symbols = loaded["notation"].get("symbols")
    symbol_names: set[str] = set()
    for symbol in symbols:
        if not isinstance(symbol, dict) or any(
            not str(symbol.get(field, "")).strip()
            for field in ("symbol", "meaning", "domain", "first_unit")
        ):
            return "notation registry contains an incomplete symbol", set(), set()
        if symbol["first_unit"] not in unit_ids:
            return (
                f"notation registry contains unknown first_unit {symbol['first_unit']}",
                set(),
                set(),
            )
        symbol_names.add(str(symbol["symbol"]))

    terms = loaded["glossary"].get("terms")
    term_names: set[str] = set()
    for term in terms:
        if not isinstance(term, dict) or any(
            not str(term.get(field, "")).strip()
            for field in ("zh", "en", "definition")
        ):
            return "glossary registry contains an incomplete term", set(), set()
        term_names.add(str(term["zh"]))
    return None, symbol_names, term_names


def validate_shared(
    manifest: dict[str, object], units: list[dict[str, object]], root: Path
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
    if not isinstance(volumes, list) or not volumes:
        return "curriculum must define at least one volume"
    volume_ids = {str(item.get("id")) for item in volumes}
    for volume in volumes:
        if not (root / str(volume.get("source", ""))).is_file():
            return f"missing volume source: {volume.get('source')}"

    identifiers = [str(unit.get("id")) for unit in units]
    if len(identifiers) != len(set(identifiers)):
        return "course graph contains duplicate unit identifiers"
    for unit in units:
        if str(unit.get("volume")) not in volume_ids:
            return f"{unit.get('id')}: unknown volume {unit.get('volume')}"

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        return "curriculum must define at least one direction track"
    track_ids = [str(track.get("id")) for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        return "curriculum contains duplicate track identifiers"
    unit_ids = set(identifiers)
    for track in tracks:
        planned_units = track.get("planned_units", [])
        if len(planned_units) != 3 or len(set(planned_units)) != 3:
            return f"{track.get('id')}: route plan must declare three unique learning units"
        for prerequisite in track.get("bridge_prerequisites", []):
            if prerequisite not in unit_ids:
                return f"{track.get('id')}: unknown bridge prerequisite {prerequisite}"
    return None

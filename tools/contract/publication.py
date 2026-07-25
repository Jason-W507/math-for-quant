from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def validate(
    manifest: dict[str, object], requested: str, root: Path
) -> tuple[str | None, list[tuple[str, int]]]:
    selected = [
        volume
        for volume in manifest.get("volumes", [])
        if requested == "all" or volume.get("id") == requested
    ]
    if not selected:
        return f"unknown volume: {requested}", []
    results: list[tuple[str, int]] = []
    for volume in selected:
        identifier = str(volume.get("id", "<unknown>"))
        relative = str(volume.get("pdf", ""))
        path = root / relative
        if not path.is_file():
            return f"{identifier}: missing publication artifact {relative}", []
        try:
            pages = len(PdfReader(path).pages)
        except Exception as error:
            return f"{identifier}: invalid publication artifact {relative}: {error}", []
        if pages < 1:
            return f"{identifier}: publication artifact has no pages {relative}", []
        results.append((identifier, pages))
    return None, results

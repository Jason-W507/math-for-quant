"""Insert generated figure wrappers after each target section's opening paragraph."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "figures" / "figure-specs.json"


def insertion_point(text: str, anchor: str) -> int:
    start = text.index(anchor) + len(anchor)
    cursor = start
    while True:
        while cursor < len(text) and text[cursor] in " \t\r\n":
            cursor += 1
        if text.startswith("\\label{", cursor):
            cursor = text.index("\n", cursor) + 1
            continue
        break
    paragraph_end = text.find("\n\n", cursor)
    if paragraph_end == -1:
        raise ValueError(f"No opening paragraph after {anchor}")
    return paragraph_end + 2


def main() -> int:
    data = json.loads(SPECS.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for spec in data["figures"]:
        grouped[(spec["file"], spec["anchor"])].append(spec["id"])

    changed = 0
    figure_ids = [spec["id"] for spec in data["figures"]]
    touched_files = {
        path.relative_to(ROOT).as_posix()
        for volume in ("upper", "lower")
        for path in (ROOT / "tex" / volume / "chapters").glob("*.tex")
    }
    for relative_file in touched_files:
        path = ROOT / relative_file
        text = path.read_text(encoding="utf-8")
        for figure_id in figure_ids:
            text = text.replace(
                f"\\input{{tex/figures/generated/{figure_id}.tex}}\n", ""
            )
        path.write_text(text, encoding="utf-8", newline="\n")

    for (relative_file, anchor), figure_ids in grouped.items():
        path = ROOT / relative_file
        text = path.read_text(encoding="utf-8")
        missing = figure_ids
        if anchor not in text:
            raise ValueError(f"Anchor not found in {relative_file}: {anchor}")
        block = "".join(
            f"\\input{{tex/figures/generated/{figure_id}.tex}}\n" for figure_id in missing
        ) + "\n"
        point = insertion_point(text, anchor)
        path.write_text(text[:point] + block + text[point:], encoding="utf-8", newline="\n")
        changed += len(missing)
    print(f"integrated {changed} generated figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jupytext


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a notebook from its canonical Jupytext source."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def semantic_cells(notebook: object) -> list[tuple[str, str]]:
    return [
        (str(cell.cell_type), str(cell.source).replace("\r\n", "\n"))
        for cell in notebook.cells
    ]


def main() -> int:
    args = parse_args()
    source = ROOT / args.source
    output = ROOT / args.output
    if not source.is_file():
        print(f"missing Jupytext source: {args.source.as_posix()}", file=sys.stderr)
        return 1

    source_notebook = jupytext.read(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    jupytext.write(source_notebook, output, fmt="ipynb")
    generated_notebook = jupytext.read(output)

    source_cells = semantic_cells(source_notebook)
    generated_cells = semantic_cells(generated_notebook)
    if source_cells != generated_cells:
        print("Jupytext roundtrip changed cell semantics", file=sys.stderr)
        return 1

    print(f"notebook={args.output.as_posix()}")
    print(f"roundtrip=passed cells={len(source_cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

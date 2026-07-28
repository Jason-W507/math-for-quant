from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "true")

import jupytext
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a notebook from its canonical Jupytext source."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def semantic_cells(notebook: object) -> list[tuple[str, str]]:
    return [
        (str(cell.cell_type), str(cell.source).replace("\r\n", "\n"))
        for cell in notebook.cells
    ]


BENIGN_ZMQ_SHUTDOWN = re.compile(
    r"^Assertion failed: Connection reset by peer \[10054\] "
    r"\(.*[\\/]signaler\.cpp:345\)\r?$"
)


def _read_kernel_stderr(stream: object) -> str:
    stream.flush()
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace")


def _replay_unexpected_kernel_stderr(stderr_text: str) -> None:
    unexpected = [
        line
        for line in stderr_text.splitlines()
        if line and BENIGN_ZMQ_SHUTDOWN.fullmatch(line) is None
    ]
    if unexpected:
        sys.stderr.write("\n".join(unexpected) + "\n")


def execute_with_isolated_kernel_stderr(notebook: object) -> object:
    """Filter one known shutdown line while preserving every real diagnostic."""
    with tempfile.TemporaryFile(mode="w+b") as kernel_stderr:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="WARNING: Insecure writes have been enabled.*",
                )
                executed = NotebookClient(
                    notebook,
                    timeout=60,
                    kernel_name="python3",
                    extra_arguments=["--log-level=ERROR"],
                    resources={"metadata": {"path": str(ROOT)}},
                ).execute(stderr=kernel_stderr)
        except BaseException:
            sys.stderr.write(_read_kernel_stderr(kernel_stderr))
            raise
        _replay_unexpected_kernel_stderr(_read_kernel_stderr(kernel_stderr))
        return executed


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
    if args.execute:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        runtime = ROOT / "build" / "jupyter-runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
        os.environ["IPYTHONDIR"] = str(ROOT / "build" / "ipython")
        executed = execute_with_isolated_kernel_stderr(generated_notebook)
        output_text = "".join(
            str(item.get("text", ""))
            for cell in executed.cells
            for item in cell.get("outputs", [])
            if item.get("output_type") == "stream"
        )
        oracle_line = next(
            (line for line in output_text.splitlines() if line.startswith("oracle=passed ")),
            None,
        )
        if oracle_line is None:
            print("executed notebook did not produce oracle evidence", file=sys.stderr)
            return 1
        jupytext.write(executed, output, fmt="ipynb")
        print(f"execution=passed {oracle_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

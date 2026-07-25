from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "curriculum" / "manifest.json"
FATAL_LOG_MARKERS = (
    "Undefined control sequence",
    "LaTeX Error:",
    "There were undefined references",
    "Missing character:",
)
MAX_OVERFULL_POINTS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two ElegantBook volumes.")
    parser.add_argument("--volume", default="all")
    return parser.parse_args()


def load_volumes() -> dict[str, dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in manifest["volumes"]}


def build_volume(volume: dict[str, str]) -> Path:
    identifier = volume["id"]
    source = Path(volume["source"])
    published_pdf = ROOT / volume["pdf"]
    job_name = published_pdf.stem
    build_directory = ROOT / "build" / "latex" / identifier
    build_directory.mkdir(parents=True, exist_ok=True)
    published_pdf.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    vendor = str(ROOT / "vendor" / "elegantbook")
    environment["TEXINPUTS"] = vendor + os.pathsep + environment.get("TEXINPUTS", "")

    command = [
        "xelatex",
        "--disable-installer",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={build_directory}",
        f"-job-name={job_name}",
        str(ROOT / source),
    ]
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise RuntimeError(f"{identifier} XeLaTeX build failed")

    log_path = build_directory / f"{job_name}.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    failures = [marker for marker in FATAL_LOG_MARKERS if marker in log]
    if failures:
        raise RuntimeError(f"{identifier} log contains: {', '.join(failures)}")
    overfull = [
        float(value)
        for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)
    ]
    severe = [value for value in overfull if value > MAX_OVERFULL_POINTS]
    if severe:
        raise RuntimeError(
            f"{identifier} log contains Overfull hbox above "
            f"{MAX_OVERFULL_POINTS:.1f}pt: max={max(severe):.3f}pt"
        )
    for value in overfull:
        print(
            f"warning: {identifier} Overfull hbox {value:.3f}pt "
            f"(allowed <= {MAX_OVERFULL_POINTS:.1f}pt)",
            file=sys.stderr,
        )

    built_pdf = build_directory / f"{job_name}.pdf"
    if not built_pdf.is_file() or built_pdf.stat().st_size == 0:
        raise RuntimeError(f"{identifier} PDF was not produced")
    shutil.copy2(built_pdf, published_pdf)
    return published_pdf


def main() -> int:
    args = parse_args()
    volumes = load_volumes()
    if args.volume != "all" and args.volume not in volumes:
        print(f"unknown volume: {args.volume}", file=sys.stderr)
        return 1
    selected = list(volumes) if args.volume == "all" else [args.volume]
    try:
        for identifier in selected:
            pdf = build_volume(volumes[identifier])
            print(f"volume={identifier} pdf={pdf.relative_to(ROOT).as_posix()}")
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

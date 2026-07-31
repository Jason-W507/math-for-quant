from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    "Too many unprocessed floats",
    "Float too large for page",
)
MAX_OVERFULL_POINTS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two ElegantBook volumes.")
    parser.add_argument("--volume", default="all")
    parser.add_argument(
        "--skip-supplements",
        action="store_true",
        help="Build only the selected main volume; useful for local layout iteration.",
    )
    return parser.parse_args()


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def attached_supplements(
    manifest: dict[str, object], selected_volume: str
) -> list[dict[str, str]]:
    return [
        item
        for item in manifest["supplements"]
        if selected_volume in item["parent_volumes"]
    ]


def release_date(environment: dict[str, str]) -> str:
    declared = environment.get("MFQ_RELEASE_DATE")
    if declared:
        datetime.strptime(declared, "%Y-%m-%d")
        return declared
    epoch = environment.get("SOURCE_DATE_EPOCH")
    if epoch:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).date().isoformat()
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cs", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    raise RuntimeError(
        "release date is unavailable; set MFQ_RELEASE_DATE or SOURCE_DATE_EPOCH, "
        "or build from a Git commit"
    )


def git_source_date_epoch() -> str | None:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ct", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def build_volume(volume: dict[str, str]) -> Path:
    identifier = volume["id"]
    source = Path(volume["source"])
    published_pdf = ROOT / volume["pdf"]
    job_name = published_pdf.stem
    build_directory = ROOT / "build" / "latex" / identifier
    build_directory.mkdir(parents=True, exist_ok=True)
    published_pdf.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    commit_epoch = git_source_date_epoch()
    if commit_epoch is not None:
        environment.setdefault("SOURCE_DATE_EPOCH", commit_epoch)
    vendor = str(ROOT / "vendor" / "elegantbook")
    environment["TEXINPUTS"] = vendor + os.pathsep + environment.get("TEXINPUTS", "")

    wrapper = build_directory / f"{job_name}-wrapper.tex"
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    wrapper.write_text(
        f"\\def\\MFQReleaseDate{{{release_date(environment)}}}\n"
        f"\\def\\MFQVersion{{{version}}}\n"
        f"\\input{{{(ROOT / source).as_posix()}}}\n",
        encoding="utf-8",
        newline="\n",
    )
    installer_option = (
        "--enable-installer"
        if environment.get("MFQ_MIKTEX_AUTO_INSTALL") == "1"
        else "--disable-installer"
    )
    command = [
        "latexmk",
        "-xelatex",
        "-bibtex",
        f"-xelatex=xelatex {installer_option} %O %S",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={build_directory}",
        f"-jobname={job_name}",
        str(wrapper),
    ]
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
        raise RuntimeError(f"{identifier} latexmk build failed")

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
    manifest = load_manifest()
    volumes = {str(item["id"]): item for item in manifest["volumes"]}
    if args.volume != "all" and args.volume not in volumes:
        print(f"unknown volume: {args.volume}", file=sys.stderr)
        return 1
    selected = list(volumes) if args.volume == "all" else [args.volume]
    built_supplements: set[str] = set()
    try:
        for identifier in selected:
            pdf = build_volume(volumes[identifier])
            print(f"volume={identifier} pdf={pdf.relative_to(ROOT).as_posix()}")
            for supplement in (
                [] if args.skip_supplements else attached_supplements(manifest, identifier)
            ):
                if supplement["id"] in built_supplements:
                    continue
                supplement_pdf = build_volume(supplement)
                built_supplements.add(supplement["id"])
                print(
                    f"supplement={supplement['id']} "
                    f"pdf={supplement_pdf.relative_to(ROOT).as_posix()}"
                )
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

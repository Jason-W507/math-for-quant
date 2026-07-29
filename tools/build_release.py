from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from contract.schema import validate_document
except ModuleNotFoundError:  # Imported as tools.build_release by tests.
    from tools.contract.schema import validate_document


ROOT = Path(__file__).resolve().parents[1]


def canonical_notebooks(root: Path = ROOT) -> list[tuple[Path, Path]]:
    sources = sorted((root / "notebooks").rglob("*.py"))
    return [
        (source, root / "output" / "notebooks" / source.relative_to(root / "notebooks").with_suffix(".ipynb"))
        for source in sources
    ]


def capstone_units(manifest: dict[str, object]) -> list[str]:
    units = list(manifest["units"])
    upper_capstone = [unit["id"] for unit in units if unit["id"] == "upper.ch17"]
    lower_capstones = [unit["id"] for unit in units if unit.get("volume") == "lower" and unit.get("published")]
    return upper_capstone + lower_capstones


def curriculum_counts(manifest: dict[str, object]) -> dict[str, int]:
    units = list(manifest["units"])
    return {
        "accepted_units": sum(unit.get("state") == "accepted" for unit in units),
        "published_units": sum(bool(unit.get("published")) for unit in units),
        "route_diagnostics": len(manifest["tracks"]),
    }


def release_asset_records(
    root: Path, assets: list[tuple[Path, str, str, str]]
) -> list[dict[str, object]]:
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    records: list[dict[str, object]] = []
    for path, kind, license_id, source in assets:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"release asset is missing or outside repository: {path}")
        content = resolved.read_bytes()
        records.append(
            {
                "path": resolved.relative_to(root.resolve()).as_posix(),
                "kind": kind,
                "license_id": license_id,
                "version": version,
                "source": source,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"release command failed ({result.returncode}): {' '.join(command)}")


def clean_generated_outputs(manifest: dict[str, object]) -> None:
    for path in (
        ROOT / "output" / "notebooks",
        ROOT / "output" / "release",
        ROOT / "build" / "latex" / "upper",
        ROOT / "build" / "latex" / "upper-solutions",
        ROOT / "build" / "latex" / "lower",
    ):
        if path.exists():
            shutil.rmtree(path)
    archive = ROOT / "output" / "math-for-quant-notebooks.zip"
    if archive.exists():
        archive.unlink()
    publications = list(manifest["volumes"]) + list(manifest["supplements"])
    for publication in publications:
        pdf = ROOT / publication["pdf"]
        if pdf.exists():
            pdf.unlink()


def build_notebooks(pairs: list[tuple[Path, Path]]) -> None:
    for source, output in pairs:
        run(
            [
                sys.executable,
                "tools/build_notebook.py",
                "--source",
                source.relative_to(ROOT).as_posix(),
                "--output",
                output.relative_to(ROOT).as_posix(),
                "--execute",
            ]
        )


def git_value(format_string: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", f"--format={format_string}", "HEAD"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("release requires a committed Git HEAD")
    return result.stdout.strip()


def write_release_manifest(
    notebooks: list[tuple[Path, Path]],
    capstones: list[str],
    counts: dict[str, int],
    curriculum: dict[str, object],
) -> Path:
    publications = list(curriculum["volumes"]) + list(curriculum["supplements"])
    pdfs = [ROOT / publication["pdf"] for publication in publications]
    data_files = sorted(path for path in (ROOT / "data").rglob("*") if path.is_file())
    archive_base = ROOT / "output" / "math-for-quant-notebooks"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", ROOT / "output" / "notebooks"))
    assets = (
        [
            (path, "pdf", "CC-BY-NC-SA-4.0", "generated from repository TeX sources")
            for path in pdfs
        ]
        + [
            (
                output,
                "executed-notebook",
                "MIT-and-CC-BY-NC-SA-4.0",
                source.relative_to(ROOT).as_posix(),
            )
            for source, output in notebooks
        ]
        + [
            (
                archive_path,
                "notebook-archive",
                "MIT-and-CC-BY-NC-SA-4.0",
                "output/notebooks",
            )
        ]
        + [
            (
                path,
                "synthetic-data",
                "CC0-1.0",
                "original synthetic repository fixture",
            )
            for path in data_files
        ]
        + [
            (
                ROOT / "vendor/elegantbook/elegantbook.cls",
                "template",
                "LPPL-1.3c",
                "ElegantBook 4.7 with documented local compatibility patch",
            ),
            (
                ROOT / "assets/cover.jpg",
                "template-asset",
                "LPPL-1.3c",
                "ElegantBook 4.7 template asset",
            ),
        ]
    )
    records = release_asset_records(ROOT, assets)
    source_date_epoch = int(git_value("%ct"))
    manifest = {
        "schema_version": 1,
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "git_commit": git_value("%H"),
        "source_date_epoch": source_date_epoch,
        "generated_at_utc": datetime.fromtimestamp(
            source_date_epoch, tz=timezone.utc
        ).replace(microsecond=0).isoformat(),
        **counts,
        "capstones": capstones,
        "license_policy": "docs/release-license-policy.md",
        "template_provenance": "docs/template-provenance.json",
        "assets": records,
    }
    schema_error = validate_document(manifest, "release-manifest")
    if schema_error is not None:
        raise ValueError(schema_error)
    output = ROOT / "output" / "release" / "release-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify the complete two-volume release.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the full unittest suite when it was already run at the same HEAD.")
    parser.add_argument(
        "--vendored-template-only",
        action="store_true",
        help="Verify committed template assets when the external baseline is unavailable in CI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((ROOT / "curriculum/manifest.json").read_text(encoding="utf-8"))
    notebooks = canonical_notebooks(ROOT)
    capstones = capstone_units(manifest)
    if len(notebooks) != 24:
        raise SystemExit(f"release requires 24 canonical notebooks, observed {len(notebooks)}")
    if len(capstones) != 7:
        raise SystemExit(f"release requires seven Capstones, observed {len(capstones)}")
    clean_generated_outputs(manifest)
    try:
        if not args.skip_tests:
            run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
        run([sys.executable, "tools/render_shared_registries.py", "--check"])
        build_notebooks(notebooks)
        run([sys.executable, "tools/build_books.py", "--volume", "all"])
        run([sys.executable, "tools/check_learning_unit.py", "--manifest", "curriculum/manifest.json", "--volume", "all"])
        if args.vendored_template_only:
            run([sys.executable, "tools/check_template_provenance.py", "--vendored-only"])
        else:
            provenance = json.loads((ROOT / "docs/template-provenance.json").read_text(encoding="utf-8"))
            run([sys.executable, "tools/check_template_provenance.py", "--source", str(provenance["external_source"])])
        release_manifest = write_release_manifest(
            notebooks, capstones, curriculum_counts(manifest), manifest
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        f"release=passed notebooks={len(notebooks)} capstones={len(capstones)} "
        f"manifest={release_manifest.relative_to(ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

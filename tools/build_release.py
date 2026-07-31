from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from contract.schema import validate_document
except ModuleNotFoundError:  # Imported as tools.build_release by tests.
    from tools.contract.schema import validate_document


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReleaseAsset:
    path: Path
    kind: str
    license_id: str
    license_files: tuple[str, ...]
    source: str
    version: str
    provenance: dict[str, str] | None = None


def canonical_notebooks(root: Path = ROOT) -> list[tuple[Path, Path]]:
    manifest = json.loads((root / "curriculum" / "manifest.json").read_text(encoding="utf-8"))
    sources = sorted(
        {
            root / unit["evidence"]["independent_oracle"]["source"]
            for unit in manifest["units"]
            if unit.get("state") == "accepted"
        }
    )
    return [
        (source, root / "output" / "notebooks" / source.relative_to(root / "notebooks").with_suffix(".ipynb"))
        for source in sources
    ]


def capstone_units(manifest: dict[str, object]) -> list[str]:
    units = list(manifest["units"])
    return [unit["id"] for unit in units if unit.get("capstone")]


def curriculum_counts(manifest: dict[str, object]) -> dict[str, int]:
    units = list(manifest["units"])
    return {
        "accepted_units": sum(unit.get("state") == "accepted" for unit in units),
        "published_units": sum(bool(unit.get("published")) for unit in units),
        "route_diagnostics": len(manifest["tracks"]),
    }


def release_asset_records(
    root: Path, assets: list[ReleaseAsset]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for asset in assets:
        resolved = asset.path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
            raise ValueError(
                f"release asset is missing or outside repository: {asset.path}"
            )
        for license_file in asset.license_files:
            if not (root / license_file).is_file():
                raise ValueError(
                    f"release asset license file is missing: {license_file}"
                )
        content = resolved.read_bytes()
        record: dict[str, object] = {
                "path": resolved.relative_to(root.resolve()).as_posix(),
                "kind": asset.kind,
                "license_id": asset.license_id,
                "license_files": list(asset.license_files),
                "version": asset.version,
                "source": asset.source,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        if asset.provenance is not None:
            record["provenance"] = dict(asset.provenance)
        records.append(record)
    return records


def registered_data_assets(root: Path = ROOT) -> list[ReleaseAsset]:
    registry_path = root / "data" / "assets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    schema_error = validate_document(registry, "data-assets")
    if schema_error is not None:
        raise ValueError(schema_error)

    registered: set[str] = set()
    license_files = {str(item["license_file"]) for item in registry["assets"]}
    assets: list[ReleaseAsset] = []
    for group in registry["assets"]:
        license_file = str(group["license_file"])
        license_text = (root / license_file).read_text(encoding="utf-8")
        if str(group["required_marker"]) not in license_text:
            raise ValueError(
                f"data license marker missing for {group['id']}: {license_file}"
            )
        for relative in group["paths"]:
            relative = str(relative)
            if relative in registered:
                raise ValueError(f"data asset is registered more than once: {relative}")
            registered.add(relative)
            assets.append(
                ReleaseAsset(
                    path=root / relative,
                    kind=str(group["kind"]),
                    license_id=str(group["license_id"]),
                    license_files=(license_file,),
                    source=str(group["source"]),
                    version=str(group["version"]),
                    provenance=(
                        {
                            key: str(group[key])
                            for key in ("schema", "selection", "missing_values", "transformations")
                        }
                        if group["kind"] == "frozen-public-data"
                        else None
                    ),
                )
            )

    excluded = {"data/assets.json", *license_files}
    observed = {
        path.relative_to(root).as_posix()
        for path in (root / "data").rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    if registered != observed:
        missing = sorted(observed - registered)
        stale = sorted(registered - observed)
        raise ValueError(
            f"data asset registry mismatch: unregistered={missing} missing={stale}"
        )
    return assets


def release_tag_error(tag: str | None, version: str) -> str | None:
    if tag is None:
        return None
    expected = f"v{version}"
    if tag != expected:
        return f"release tag {tag!r} does not match VERSION {expected!r}"
    return None


def ensure_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("release could not inspect the Git worktree")
    if result.stdout.strip():
        raise RuntimeError(
            "release requires a clean Git worktree; commit or remove source changes"
        )


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
        ROOT / "build" / "latex" / "solutions",
        ROOT / "build" / "latex" / "lower",
    ):
        if path.exists():
            shutil.rmtree(path)
    archive = ROOT / "output" / "math-for-quant-notebooks.zip"
    if archive.exists():
        archive.unlink()
    legacy_upper_solutions = (
        ROOT / "output" / "pdf" / "math-for-quant-upper-solutions.pdf"
    )
    if legacy_upper_solutions.exists():
        legacy_upper_solutions.unlink()
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


def build_notebook_archive(
    notebooks: list[tuple[Path, Path]],
    source_date_epoch: int,
    root: Path = ROOT,
    archive_path: Path | None = None,
) -> Path:
    archive_path = archive_path or root / "output" / "math-for-quant-notebooks.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromtimestamp(
        max(source_date_epoch, 315532800), tz=timezone.utc
    ).timetuple()[:6]
    entries = [
        (
            output,
            "notebooks/"
            + output.relative_to(root / "output" / "notebooks").as_posix(),
        )
        for _, output in notebooks
    ] + [
        (root / "LICENSE", "LICENSE"),
        (root / "LICENSE-CONTENT.md", "LICENSE-CONTENT.md"),
        (
            root / "docs" / "release-license-policy.md",
            "docs/release-license-policy.md",
        ),
    ]
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, member in sorted(entries, key=lambda item: item[1]):
            if not source.is_file():
                raise ValueError(f"notebook archive member is missing: {source}")
            info = zipfile.ZipInfo(member, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return archive_path


def write_release_manifest(
    notebooks: list[tuple[Path, Path]],
    capstones: list[str],
    counts: dict[str, int],
    curriculum: dict[str, object],
) -> Path:
    release_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    publications = list(curriculum["volumes"]) + list(curriculum["supplements"])
    pdfs = [ROOT / publication["pdf"] for publication in publications]
    source_date_epoch = int(git_value("%ct"))
    archive_path = build_notebook_archive(notebooks, source_date_epoch)
    assets = (
        [
            ReleaseAsset(
                path=path,
                kind="pdf",
                license_id="CC-BY-NC-SA-4.0",
                license_files=("LICENSE-CONTENT.md",),
                source="generated from repository TeX sources",
                version=release_version,
            )
            for path in pdfs
        ]
        + [
            ReleaseAsset(
                path=output,
                kind="executed-notebook",
                license_id="MIT-and-CC-BY-NC-SA-4.0",
                license_files=("LICENSE", "LICENSE-CONTENT.md"),
                source=source.relative_to(ROOT).as_posix(),
                version=release_version,
            )
            for source, output in notebooks
        ]
        + [
            ReleaseAsset(
                path=archive_path,
                kind="notebook-archive",
                license_id="MIT-and-CC-BY-NC-SA-4.0",
                license_files=(
                    "LICENSE",
                    "LICENSE-CONTENT.md",
                    "docs/release-license-policy.md",
                ),
                source="output/notebooks",
                version=release_version,
            )
        ]
        + registered_data_assets(ROOT)
        + [
            ReleaseAsset(
                path=ROOT / "vendor/elegantbook/elegantbook.cls",
                kind="template",
                license_id="LPPL-1.3c",
                license_files=("vendor/elegantbook/LPPL-License.txt",),
                source="ElegantBook 4.7 with documented local compatibility patch",
                version="ElegantBook-4.7",
            ),
            ReleaseAsset(
                path=ROOT / "assets/cover.jpg",
                kind="template-asset",
                license_id="LPPL-1.3c",
                license_files=("vendor/elegantbook/LPPL-License.txt",),
                source="ElegantBook 4.7 template asset",
                version="ElegantBook-4.7",
            ),
            ReleaseAsset(
                path=ROOT / "data/assets.json",
                kind="data-registry",
                license_id="CC0-1.0",
                license_files=("data/README.md",),
                source="repository data provenance registry",
                version=release_version,
            ),
            ReleaseAsset(
                path=ROOT / "docs/release-license-policy.md",
                kind="license-policy",
                license_id="CC-BY-NC-SA-4.0",
                license_files=("LICENSE-CONTENT.md",),
                source="repository release policy",
                version=release_version,
            ),
        ]
    )
    records = release_asset_records(ROOT, assets)
    manifest = {
        "schema_version": 1,
        "version": release_version,
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
    try:
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        tag_error = release_tag_error(os.environ.get("MFQ_RELEASE_TAG"), version)
        if tag_error is not None:
            raise ValueError(tag_error)
        ensure_clean_worktree()
        clean_generated_outputs(manifest)
        if not args.skip_tests:
            run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
        run([sys.executable, "tools/render_shared_registries.py", "--check"])
        build_notebooks(notebooks)
        run([sys.executable, "tools/build_books.py", "--volume", "all"])
        run([sys.executable, "tools/check_pdf_visual_regression.py"])
        run([sys.executable, "tools/check_learning_unit.py", "--manifest", "curriculum/manifest.json", "--track", "all"])
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

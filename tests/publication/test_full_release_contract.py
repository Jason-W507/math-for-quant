from __future__ import annotations

import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import tools.build_release as build_release
from tools.build_release import (
    ReleaseAsset,
    build_notebook_archive,
    canonical_notebooks,
    capstone_units,
    clean_generated_outputs,
    curriculum_counts,
    registered_data_assets,
    release_asset_records,
    release_tag_error,
)
from tools.contract.schema import validate_document


ROOT = Path(__file__).resolve().parents[2]


class FullReleaseContractTests(unittest.TestCase):
    def test_release_cleanup_removes_the_legacy_upper_only_solutions_pdf(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum/manifest.json").read_text(encoding="utf-8")
        )
        test_root = ROOT / "build" / "test-release" / "cleanup-root"
        legacy = test_root / "output" / "pdf" / "math-for-quant-upper-solutions.pdf"

        with (
            mock.patch.object(build_release, "ROOT", test_root),
            mock.patch.object(
                Path, "exists", autospec=True, side_effect=lambda path: path == legacy
            ),
            mock.patch.object(Path, "unlink", autospec=True) as unlink,
        ):
            clean_generated_outputs(manifest)

        unlink.assert_called_once_with(legacy)

    def test_all_canonical_notebooks_have_unique_release_paths(self) -> None:
        pairs = canonical_notebooks(ROOT)
        self.assertGreater(len(pairs), 0)
        self.assertEqual(len({output for _, output in pairs}), len(pairs))
        self.assertTrue(all(source.suffix == ".py" for source, _ in pairs))
        self.assertTrue(all(output.suffix == ".ipynb" for _, output in pairs))

    def test_lower_notebook_ignores_ipykernel_connection_arguments(self) -> None:
        output = ROOT / "build" / "notebooks" / "release-lower-ch01.ipynb"
        result = subprocess.run(
            [
                sys.executable,
                "tools/build_notebook.py",
                "--source",
                "notebooks/lower/ch01_multifactor.py",
                "--output",
                output.relative_to(ROOT).as_posix(),
                "--execute",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("execution=passed oracle=passed", result.stdout)
        notebook = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 2)
        self.assertNotIn(
            "sys.argv = [sys.argv[0]]",
            "\n".join(str(cell["source"]) for cell in notebook["cells"]),
        )

    def test_seven_capstones_are_derived_from_the_curriculum(self) -> None:
        manifest = json.loads((ROOT / "curriculum/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(capstone_units(manifest), ["upper.ch17", "lower.ch01", "lower.ch02", "lower.ch03", "lower.ch04", "lower.ch05", "lower.ch06"])

    def test_release_driver_does_not_hardcode_asset_counts(self) -> None:
        source = (ROOT / "tools" / "build_release.py").read_text(encoding="utf-8")
        self.assertNotIn("len(notebooks) != 24", source)
        self.assertNotIn("len(capstones) != 7", source)

    def test_release_assets_have_checksum_version_and_license(self) -> None:
        fixture = ROOT / "build" / "test-release" / "artifact.txt"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("release-evidence\n", encoding="utf-8")
        records = release_asset_records(
            ROOT,
            [
                ReleaseAsset(
                    path=fixture,
                    kind="test",
                    license_id="MIT",
                    license_files=("LICENSE",),
                    source="repository",
                    version="0.2.0",
                )
            ],
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["path"], fixture.relative_to(ROOT).as_posix())
        self.assertEqual(record["kind"], "test")
        self.assertEqual(record["license_id"], "MIT")
        self.assertEqual(record["license_files"], ["LICENSE"])
        self.assertEqual(record["version"], "0.2.0")
        self.assertEqual(record["source"], "repository")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertGreater(record["bytes"], 0)

    def test_registered_data_assets_cover_every_payload(self) -> None:
        assets = registered_data_assets(ROOT)
        self.assertEqual(len(assets), 25)
        self.assertTrue(all(asset.license_id == "CC0-1.0" for asset in assets))

    def test_release_tag_must_match_version(self) -> None:
        self.assertIsNone(release_tag_error(None, "0.2.0"))
        self.assertIsNone(release_tag_error("v0.2.0", "0.2.0"))
        self.assertEqual(
            release_tag_error("v0.3.0", "0.2.0"),
            "release tag 'v0.3.0' does not match VERSION 'v0.2.0'",
        )

    def test_notebook_archive_carries_both_licenses_and_policy(self) -> None:
        test_root = ROOT / "build" / "test-release" / "archive-root"
        output = test_root / "output" / "notebooks" / "test-release.ipynb"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        (test_root / "docs").mkdir(parents=True, exist_ok=True)
        (test_root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
        (test_root / "LICENSE-CONTENT.md").write_text(
            "CC BY-NC-SA 4.0\n", encoding="utf-8"
        )
        (test_root / "docs" / "release-license-policy.md").write_text(
            "license map\n", encoding="utf-8"
        )
        archive = ROOT / "build" / "test-release" / "notebooks.zip"
        build_notebook_archive(
            [(test_root / "canonical.py", output)],
            315532800,
            root=test_root,
            archive_path=archive,
        )
        with zipfile.ZipFile(archive) as package:
            self.assertEqual(
                set(package.namelist()),
                {
                    "LICENSE",
                    "LICENSE-CONTENT.md",
                    "docs/release-license-policy.md",
                    "notebooks/test-release.ipynb",
                },
            )

    def test_curriculum_counts_are_derived_from_the_manifest(self) -> None:
        manifest = json.loads((ROOT / "curriculum/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            curriculum_counts(manifest),
            {"accepted_units": 24, "published_units": 23, "route_diagnostics": 6},
        )

    def test_release_manifest_schema_accepts_the_public_shape(self) -> None:
        document = {
            "schema_version": 1,
            "version": "0.2.0",
            "git_commit": "a" * 40,
            "source_date_epoch": 1,
            "generated_at_utc": "1970-01-01T00:00:01+00:00",
            "accepted_units": 24,
            "published_units": 23,
            "route_diagnostics": 6,
            "capstones": [
                "upper.ch17",
                "lower.ch01",
                "lower.ch02",
                "lower.ch03",
                "lower.ch04",
                "lower.ch05",
                "lower.ch06",
            ],
            "license_policy": "docs/release-license-policy.md",
            "template_provenance": "docs/template-provenance.json",
            "assets": [
                {
                    "path": "artifact.txt",
                    "kind": "test",
                    "license_id": "MIT",
                    "license_files": ["LICENSE"],
                    "version": "0.2.0",
                    "source": "repository",
                    "bytes": 1,
                    "sha256": "b" * 64,
                }
            ],
        }
        self.assertIsNone(validate_document(document, "release-manifest"))


if __name__ == "__main__":
    unittest.main()

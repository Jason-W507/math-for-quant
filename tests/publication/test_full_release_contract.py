from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.build_release import (
    canonical_notebooks,
    capstone_units,
    curriculum_counts,
    release_asset_records,
)
from tools.contract.schema import validate_document


ROOT = Path(__file__).resolve().parents[2]


class FullReleaseContractTests(unittest.TestCase):
    def test_all_canonical_notebooks_have_unique_release_paths(self) -> None:
        pairs = canonical_notebooks(ROOT)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(len({output for _, output in pairs}), 24)
        self.assertTrue(all(source.suffix == ".py" for source, _ in pairs))
        self.assertTrue(all(output.suffix == ".ipynb" for _, output in pairs))

    def test_seven_capstones_are_derived_from_the_curriculum(self) -> None:
        manifest = json.loads((ROOT / "curriculum/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(capstone_units(manifest), ["upper.ch17", "lower.ch01", "lower.ch02", "lower.ch03", "lower.ch04", "lower.ch05", "lower.ch06"])

    def test_release_assets_have_checksum_version_and_license(self) -> None:
        fixture = ROOT / "build" / "test-release" / "artifact.txt"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("release-evidence\n", encoding="utf-8")
        records = release_asset_records(
            ROOT, [(fixture, "test", "MIT", "repository")]
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["path"], fixture.relative_to(ROOT).as_posix())
        self.assertEqual(record["kind"], "test")
        self.assertEqual(record["license_id"], "MIT")
        self.assertEqual(record["version"], "0.2.0")
        self.assertEqual(record["source"], "repository")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertGreater(record["bytes"], 0)

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

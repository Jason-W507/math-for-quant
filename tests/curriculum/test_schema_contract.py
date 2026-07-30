from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from tools.contract.schema import validate_document
from tools.check_learning_unit import selected_track_units


ROOT = Path(__file__).resolve().parents[2]


class SchemaDrivenCurriculumTests(unittest.TestCase):
    def run_contract(self, manifest: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "check_learning_unit.py"),
                "--manifest",
                str(manifest),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def fixture(self, name: str, mutate: object) -> Path:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        mutate(manifest)
        path = ROOT / "build" / "test-manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_manifest_shape_is_checked_by_json_schema(self) -> None:
        path = self.fixture(
            "schema-missing-volume-title.json",
            lambda manifest: manifest["volumes"][0].pop("title"),
        )
        result = self.run_contract(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest schema validation failed", result.stderr)

    def test_manifest_rejects_a_non_array_units_field_without_traceback(self) -> None:
        path = self.fixture(
            "schema-null-units.json",
            lambda manifest: manifest.__setitem__("units", None),
        )
        result = self.run_contract(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest schema validation failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_reading_routes_are_closed_under_direct_prerequisites(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        prerequisites = {
            unit["id"]: set(unit.get("prerequisites", []))
            for unit in manifest["units"]
        }
        for route in manifest["reading_routes"]:
            seen: set[str] = set()
            for unit_id in route["units"]:
                with self.subTest(route=route["id"], unit=unit_id):
                    self.assertLessEqual(prerequisites[unit_id], seen)
                seen.add(unit_id)

    def test_curriculum_summary_uses_manifest_counts_not_constants(self) -> None:
        path = self.fixture(
            "one-track.json",
            lambda manifest: manifest.__setitem__("tracks", manifest["tracks"][:1]),
        )
        result = self.run_contract(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("volumes=2 upper_chapters=17 tracks=1", result.stdout)

    def test_all_public_json_documents_have_declared_schemas(self) -> None:
        for name in (
            "manifest",
            "notation",
            "glossary",
            "oracle",
            "licenses",
            "data-assets",
            "release-manifest",
        ):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "schemas" / f"{name}.schema.json").is_file())

        licenses = json.loads(
            (ROOT / "evidence" / "ch17" / "licenses.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIsNone(validate_document(licenses, "licenses"))

    def test_track_seam_rejects_a_route_before_three_units_are_accepted(self) -> None:
        path = self.fixture(
            "multifactor-one-draft.json",
            lambda manifest: next(
                unit
                for unit in manifest["units"]
                if unit["id"] == "lower.multifactor.estimation"
            ).__setitem__("state", "draft"),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "check_learning_unit.py"),
                "--manifest",
                str(path),
                "--track",
                "multifactor",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("three accepted learning units", result.stderr)

    def test_track_plan_requires_three_declared_units(self) -> None:
        path = self.fixture(
            "track-with-two-units.json",
            lambda manifest: manifest["tracks"][0].__setitem__(
                "planned_units", manifest["tracks"][0]["planned_units"][:2]
            ),
        )
        result = self.run_contract(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest schema validation failed", result.stderr)

    def test_route_unit_schema_requires_the_v03_evidence_contract(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        route_unit = json.loads(json.dumps(manifest["units"][1]))
        route_unit.update(
            {
                "id": "lower.multifactor.model",
                "volume": "lower",
                "track": "multifactor",
                "track_stage": "model-math",
            }
        )
        manifest["units"].append(route_unit)

        error = validate_document(manifest, "manifest")

        self.assertIsNotNone(error)
        self.assertIn("dual_track_data", error)

    def test_route_selection_rejects_non_route_and_cross_route_drift(self) -> None:
        track = {
            "id": "multifactor",
            "planned_units": [
                "lower.multifactor.model",
                "lower.multifactor.estimation",
                "lower.multifactor.research",
            ],
        }
        units = [
            {
                "id": identifier,
                "state": "accepted",
                "published": True,
                "track": "multifactor",
                "track_stage": stage,
            }
            for identifier, stage in zip(
                track["planned_units"],
                ("model-math", "estimation-numerics", "oos-frictions-capstone"),
            )
        ]
        _, error = selected_track_units(track, units)
        self.assertIsNone(error)

        units[0].pop("track")
        _, error = selected_track_units(track, units)
        self.assertIn("exactly three", error)
        units[0]["track"] = "stat-arb"
        _, error = selected_track_units(track, units)
        self.assertIn("exactly three", error)


if __name__ == "__main__":
    unittest.main()

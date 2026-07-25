from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LearningUnitContractTests(unittest.TestCase):
    def run_contract(self, fixture: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "check_learning_unit.py"),
                "--manifest",
                str(ROOT / fixture),
                *extra_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_rejects_an_accepted_unit_without_derivation_evidence(self) -> None:
        result = self.run_contract(
            "tests/fixtures/incomplete-unit.json",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "foundation.incomplete: missing evidence field core_derivation",
            result.stderr,
        )

    def test_rejects_every_missing_evidence_field(self) -> None:
        required_fields = (
            "notation_and_assumptions",
            "core_derivation",
            "independent_oracle",
            "questions",
            "hints",
            "solutions",
            "capstone_connection",
        )
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        accepted = next(
            unit for unit in manifest["units"] if unit["id"] == "foundation.oracle-smoke"
        )
        fixture_directory = ROOT / "build" / "test-manifests"
        fixture_directory.mkdir(parents=True, exist_ok=True)

        for field in required_fields:
            with self.subTest(field=field):
                fixture = json.loads(json.dumps(accepted))
                del fixture["evidence"][field]
                fixture_path = fixture_directory / f"missing-{field}.json"
                fixture_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "question_levels": [
                                "oral",
                                "derivation",
                                "computation",
                                "research",
                            ],
                            "units": [fixture],
                        }
                    ),
                    encoding="utf-8",
                )
                result = self.run_contract(
                    fixture_path.relative_to(ROOT).as_posix(),
                    "--unit",
                    "foundation.oracle-smoke",
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    result.stderr,
                    f"foundation.oracle-smoke: missing evidence field {field}\n",
                )

    def test_accepts_a_complete_unit_with_an_independent_oracle(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "foundation.oracle-smoke",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=foundation.oracle-smoke\n"
            "evidence=4/4\n"
            "oracle=passed observed=0.006666666667 "
            "expected=0.006666666667\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_rejects_a_prerequisite_that_is_not_in_the_course_graph(self) -> None:
        result = self.run_contract("tests/fixtures/invalid-prerequisite.json")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "upper.ch02: unknown prerequisite upper.missing\n",
        )

    def test_rejects_a_cycle_in_the_course_graph(self) -> None:
        result = self.run_contract("tests/fixtures/cyclic-prerequisite.json")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "course graph contains a prerequisite cycle at upper.a\n",
        )

    def test_jupytext_source_generates_a_semantically_equal_notebook(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/foundation/independent_oracle.py",
                "--output",
                "build/notebooks/foundation/independent_oracle.ipynb",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "notebook=build/notebooks/foundation/independent_oracle.ipynb\n"
            "roundtrip=passed cells=2\n",
        )
        self.assertEqual(result.stderr, "")
        self.assertTrue(
            (ROOT / "build/notebooks/foundation/independent_oracle.ipynb").is_file()
        )

    def test_external_elegantbook_template_matches_the_recorded_baseline(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "check_template_provenance.py"),
                "--source",
                "D:/Latex/ElegantBook",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "template=ElegantBook-4.7\n"
            "external-baseline=passed files=3\n"
            "vendored-assets=passed files=2\n",
        )
        self.assertEqual(result.stderr, "")

    def test_main_manifest_exposes_the_shared_curriculum_contract(self) -> None:
        result = self.run_contract("curriculum/manifest.json")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "curriculum=passed volumes=2 upper_chapters=17 tracks=6\n"
            "question-levels=passed count=4\n"
            "registries=passed count=2\n"
            "course-graph=passed units=18\n"
            "accepted-units=1\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()

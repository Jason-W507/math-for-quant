from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from pypdf import PdfWriter


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
            "evidence=7/7\n"
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
                "--execute",
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
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed observed=0.006666666667 "
            "expected=0.006666666667\n",
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
            "accepted-units=2\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_volume_scope_rejects_a_missing_publication_artifact(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["volumes"][0]["pdf"] = "build/test-publication/missing-upper.pdf"
        fixture_path = ROOT / "build" / "test-manifests" / "missing-publication.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture_path.relative_to(ROOT).as_posix(),
            "--volume",
            "upper",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "upper: missing publication artifact "
            "build/test-publication/missing-upper.pdf\n",
        )

    def test_volume_scope_accepts_a_parseable_publication_artifact(self) -> None:
        publication = ROOT / "build" / "test-publication" / "upper.pdf"
        publication.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with publication.open("wb") as stream:
            writer.write(stream)

        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["volumes"][0]["pdf"] = publication.relative_to(ROOT).as_posix()
        fixture_path = ROOT / "build" / "test-manifests" / "valid-publication.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture_path.relative_to(ROOT).as_posix(),
            "--volume",
            "upper",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("publication=passed volume=upper pages=1\n", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_rejects_an_empty_evidence_artifact(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        empty = ROOT / "build" / "test-evidence" / "empty-derivation.md"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_text("", encoding="utf-8")
        unit = next(
            item for item in manifest["units"] if item["id"] == "foundation.oracle-smoke"
        )
        unit["evidence"]["core_derivation"] = empty.relative_to(ROOT).as_posix()
        fixture = ROOT / "build" / "test-manifests" / "empty-evidence.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture.relative_to(ROOT).as_posix(),
            "--unit",
            "foundation.oracle-smoke",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "foundation.oracle-smoke: empty evidence artifact "
            "build/test-evidence/empty-derivation.md\n",
        )

    def test_rejects_questions_without_all_four_public_levels(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        questions = ROOT / "build" / "test-evidence" / "incomplete-questions.md"
        questions.parent.mkdir(parents=True, exist_ok=True)
        questions.write_text("# 问题\n\n1. 只有一个普通问题。\n", encoding="utf-8")
        unit = next(
            item for item in manifest["units"] if item["id"] == "foundation.oracle-smoke"
        )
        unit["evidence"]["questions"] = questions.relative_to(ROOT).as_posix()
        fixture = ROOT / "build" / "test-manifests" / "incomplete-questions.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture.relative_to(ROOT).as_posix(),
            "--unit",
            "foundation.oracle-smoke",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "foundation.oracle-smoke: questions must include 口述概念, 笔试推导, "
            "数值编程, 研究判断\n",
        )

    def test_rejects_a_structurally_empty_notation_registry(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        registry = ROOT / "build" / "test-evidence" / "empty-notation.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(
            json.dumps({"schema_version": 1, "symbols": []}), encoding="utf-8"
        )
        manifest["registries"]["notation"] = registry.relative_to(ROOT).as_posix()
        fixture = ROOT / "build" / "test-manifests" / "empty-notation.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(fixture.relative_to(ROOT).as_posix())

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "notation registry must contain symbols\n")

    def test_rejects_a_unit_symbol_missing_from_the_shared_registry(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        unit = next(
            item for item in manifest["units"] if item["id"] == "foundation.oracle-smoke"
        )
        unit["evidence"]["notation_symbols"] = ["x_t"]
        fixture = ROOT / "build" / "test-manifests" / "unregistered-symbol.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture.relative_to(ROOT).as_posix(),
            "--unit",
            "foundation.oracle-smoke",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "foundation.oracle-smoke: unregistered notation symbol x_t\n",
        )

    def test_accepts_chapter_one_with_bound_and_counterexample_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch01",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch01\n"
            "evidence=7/7\n"
            "oracle=passed weighted_return=0.013000 lower=-0.010000 "
            "upper=0.030000 counterexample=0.035000\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_rejects_solution_without_published_oracle_markers(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        unit = next(item for item in manifest["units"] if item["id"] == "upper.ch01")
        oracle = json.loads(
            (ROOT / "evidence" / "ch01" / "oracle.json").read_text(encoding="utf-8")
        )
        oracle["published_markers"] = ["oracle-only-marker"]
        oracle_path = ROOT / "build" / "test-evidence" / "missing-answer-marker.json"
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        unit["evidence"]["independent_oracle"]["oracle"] = oracle_path.relative_to(
            ROOT
        ).as_posix()
        fixture = ROOT / "build" / "test-manifests" / "missing-answer-marker.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture.relative_to(ROOT).as_posix(), "--unit", "upper.ch01"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "upper.ch01: solutions missing published oracle marker oracle-only-marker\n",
        )

    def test_rejects_detached_feedback_for_a_published_unit(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        detached = ROOT / "build" / "test-evidence" / "detached-questions.md"
        detached.parent.mkdir(parents=True, exist_ok=True)
        detached.write_text(
            "# 四级问题组\n\n"
            "1. 口述概念\n2. 笔试推导\n3. 数值编程\n4. 研究判断\n",
            encoding="utf-8",
        )
        unit = next(item for item in manifest["units"] if item["id"] == "upper.ch01")
        unit["evidence"]["questions"] = detached.relative_to(ROOT).as_posix()
        fixture = ROOT / "build" / "test-manifests" / "detached-feedback.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(manifest), encoding="utf-8")

        result = self.run_contract(
            fixture.relative_to(ROOT).as_posix(), "--unit", "upper.ch01"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "upper.ch01: published evidence field questions must reference tex source\n",
        )


if __name__ == "__main__":
    unittest.main()

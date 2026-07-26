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
            "accepted-units=13\n"
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

    def test_chapter_two_notebook_reproduces_contraction_and_nonuniform_oracles(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch02_analysis_foundations.py",
                "--output",
                "build/notebooks/upper/ch02_analysis_foundations.ipynb",
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
            "notebook=build/notebooks/upper/ch02_analysis_foundations.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed contraction_x5=0.067232 "
            "error=0.032768 bound=0.032768 "
            "witness10=0.500000 witness100=0.500000\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_two_with_contraction_and_nonuniform_oracles(
        self,
    ) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch02",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch02\n"
            "evidence=7/7\n"
            "oracle=passed contraction_x5=0.067232 error=0.032768 "
            "bound=0.032768 witness10=0.500000 witness100=0.500000\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_three_notebook_reproduces_simple_and_spike_oracles(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch03_measure_integration.py",
                "--output",
                "build/notebooks/upper/ch03_measure_integration.ipynb",
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
            "notebook=build/notebooks/upper/ch03_measure_integration.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed simple_m2=0.375000 "
            "simple_m4=0.468750 simple_m8=0.498047 "
            "spike10=1.000000 spike100=1.000000 "
            "point10=0.000000 point100=0.000000 gap=1.000000\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_three_with_simple_and_spike_oracles(self) -> None:
        oracle = json.loads(
            (ROOT / "evidence" / "ch03" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            oracle["published_markers"],
            [
                "0.375",
                "0.46875",
                "0.498046875",
                "n(1/n)=1",
                "点值为零",
                "极限交换差额为 1",
            ],
        )
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch03",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch03\n"
            "evidence=7/7\n"
            "oracle=passed simple_m2=0.375000 simple_m4=0.468750 "
            "simple_m8=0.498047 spike10=1.000000 spike100=1.000000 "
            "point10=0.000000 point100=0.000000 gap=1.000000\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_four_notebook_reproduces_lp_and_fubini_oracles(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch04_lp_product_measure.py",
                "--output",
                "build/notebooks/upper/ch04_lp_product_measure.ipynb",
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
            "notebook=build/notebooks/upper/ch04_lp_product_measure.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed lp1=0.500000 lp2=0.577350 "
            "lp4=0.668740 midpoint=0.333312988 error=0.000020345 "
            "bound=0.000020345 row_first=1 col_first=0 square10=1 "
            "rectangle10=0 abs10=19 square100=1 rectangle100=0 abs100=199\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_four_with_lp_and_fubini_oracles(self) -> None:
        oracle = json.loads(
            (ROOT / "evidence" / "ch04" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            oracle["published_markers"],
            [
                "0.577350",
                "0.333312988",
                "0.000020345",
                "先逐行求和得到 1",
                "先逐列求和得到 0",
                "2N-1",
            ],
        )
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch04",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch04\n"
            "evidence=7/7\n"
            "oracle=passed lp1=0.500000 lp2=0.577350 lp4=0.668740 "
            "midpoint=0.333312988 error=0.000020345 bound=0.000020345 "
            "row_first=1 col_first=0 square10=1 rectangle10=0 abs10=19 "
            "square100=1 rectangle100=0 abs100=199\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_five_notebook_cross_checks_matrix_derivatives(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch05_matrix_calculus.py",
                "--output",
                "build/notebooks/upper/ch05_matrix_calculus.ipynb",
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
            "notebook=build/notebooks/upper/ch05_matrix_calculus.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed value=-1.000000 "
            "gradient=(0.000000,-0.500000) chain=(4.000000,-6.500000) "
            "max_error=2.620e-11 left=-1.0 right=1.0\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_five_with_derivative_cross_checks(self) -> None:
        oracle = json.loads(
            (ROOT / "evidence" / "ch05" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            oracle["published_markers"],
            [
                "梯度为 $(0,-0.5)^T$",
                "链式梯度为 $(4,-6.5)^T$",
                "最大差异小于 $10^{-9}$",
                "左差商为 $-1$",
                "右差商为 $1$",
            ],
        )
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch05",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch05\n"
            "evidence=7/7\n"
            "oracle=passed value=-1.000000 gradient=(0.000000,-0.500000) "
            "chain=(4.000000,-6.500000) max_error=2.620e-11 "
            "left=-1.0 right=1.0\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_six_notebook_reproduces_projection_and_conditioning(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch06_linear_algebra.py",
                "--output",
                "build/notebooks/upper/ch06_linear_algebra.ipynb",
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
            "notebook=build/notebooks/upper/ch06_linear_algebra.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed beta=(1.166667,0.500000) "
            "sse=0.166667 orth_error=6.661e-16 recon_error=3.331e-16 "
            "sigma=(2.676243,0.915272) condition=4.000e+06 "
            "amplification=1000001 relative=1.000001\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_six_with_projection_and_conditioning(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch06",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch06\n"
            "evidence=7/7\n"
            "oracle=passed beta=(1.166667,0.500000) sse=0.166667 "
            "orth_error=6.661e-16 recon_error=3.331e-16 "
            "sigma=(2.676243,0.915272) condition=4.000e+06 "
            "amplification=1000001 relative=1.000001\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_seven_notebook_reproduces_distribution_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch07_probability_distributions.py",
                "--output",
                "build/notebooks/upper/ch07_probability_distributions.ipynb",
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
            "notebook=build/notebooks/upper/ch07_probability_distributions.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed mean=0.250000 variance=1.187500 "
            "joint_mass=1.000000 marginals=(0.500000,0.500000) "
            "bernoulli=(0.250000,0.187500,1.250000) "
            "poisson_pgf=0.367879 normal_mgf=1.133148 "
            "pareto_truncated=(2.302585,6.907755)\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_seven_rejects_negative_probability_mass(self) -> None:
        oracle = json.loads(
            (ROOT / "evidence" / "ch07" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        oracle["finite_probabilities"] = [-0.25, 0.75, 0.5]
        oracle["expected_mean"] = 1.25
        oracle["expected_variance"] = 0.1875
        fixture = ROOT / "build" / "test-fixtures" / "ch07-negative-mass.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(oracle), encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "notebooks" / "upper" / "ch07_probability_distributions.py"),
                    str(fixture),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            fixture.unlink(missing_ok=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonnegative", result.stderr)

    def test_accepts_chapter_seven_with_distribution_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch07",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch07\n"
            "evidence=7/7\n"
            "oracle=passed mean=0.250000 variance=1.187500 "
            "joint_mass=1.000000 marginals=(0.500000,0.500000) "
            "bernoulli=(0.250000,0.187500,1.250000) "
            "poisson_pgf=0.367879 normal_mgf=1.133148 "
            "pareto_truncated=(2.302585,6.907755)\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_eight_notebook_reproduces_conditioning_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch08_conditioning.py",
                "--output",
                "build/notebooks/upper/ch08_conditioning.ipynb",
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
            "notebook=build/notebooks/upper/ch08_conditioning.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed conditional=(1.000000,3.000000) "
            "tower=2.000000 kernel_rows=(1.000000,1.000000) "
            "mixture=(0.250000,0.500000,0.250000) "
            "marginal_joint=0.500000 marginal_product=0.250000 "
            "conditional_error=0.000e+00\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_eight_rejects_signed_kernel_mixing_weights(self) -> None:
        oracle = json.loads(
            (ROOT / "evidence" / "ch08" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        oracle["conditioning_probabilities"] = [-0.5, 1.5]
        oracle["expected_mixture"] = [-0.25, 0.5, 0.75]
        fixture = ROOT / "build" / "test-fixtures" / "ch08-signed-weights.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(oracle), encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "notebooks" / "upper" / "ch08_conditioning.py"),
                    str(fixture),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            fixture.unlink(missing_ok=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixing weights", result.stderr)

    def test_accepts_chapter_eight_with_conditioning_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch08",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch08\n"
            "evidence=7/7\n"
            "oracle=passed conditional=(1.000000,3.000000) "
            "tower=2.000000 kernel_rows=(1.000000,1.000000) "
            "mixture=(0.250000,0.500000,0.250000) "
            "marginal_joint=0.500000 marginal_product=0.250000 "
            "conditional_error=0.000e+00\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_nine_notebook_reproduces_limit_theorem_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch09_limit_theorems.py",
                "--output",
                "build/notebooks/upper/ch09_limit_theorems.ipynb",
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
            "notebook=build/notebooks/upper/ch09_limit_theorems.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed mean=0.299794 theory_sd=0.032404 "
            "observed_sd=0.032672 clt_coverage=0.943100 exact_tail=0.002565 "
            "simulated_tail=0.002850 hoeffding=0.036631 "
            "cauchy_medians=(0.980865,1.029577)\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_nine_with_limit_theorem_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch09",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch09\n"
            "evidence=7/7\n"
            "oracle=passed mean=0.299794 theory_sd=0.032404 "
            "observed_sd=0.032672 clt_coverage=0.943100 exact_tail=0.002565 "
            "simulated_tail=0.002850 hoeffding=0.036631 "
            "cauchy_medians=(0.980865,1.029577)\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_ten_notebook_reproduces_inference_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch10_statistical_inference.py",
                "--output",
                "build/notebooks/upper/ch10_statistical_inference.ipynb",
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
            "notebook=build/notebooks/upper/ch10_statistical_inference.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed mean=0.399938 theory_var=0.000960 "
            "empirical_var=0.000966 clt_coverage=0.954700 slope=1.496775 "
            "true_se=0.393283 empirical_se=0.394608 "
            "naive_coverage=0.868900 robust_coverage=0.938900 "
            "naive_fwer=0.639975 bonferroni_fwer=0.047400\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_ten_with_inference_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch10",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch10\n"
            "evidence=7/7\n"
            "oracle=passed mean=0.399938 theory_var=0.000960 "
            "empirical_var=0.000966 clt_coverage=0.954700 slope=1.496775 "
            "true_se=0.393283 empirical_se=0.394608 "
            "naive_coverage=0.868900 robust_coverage=0.938900 "
            "naive_fwer=0.639975 bonferroni_fwer=0.047400\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_eleven_notebook_reproduces_process_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch11_stochastic_processes.py",
                "--output",
                "build/notebooks/upper/ch11_stochastic_processes.ipynb",
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
            "notebook=build/notebooks/upper/ch11_stochastic_processes.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed "
            "markov_p5=(0.612500,0.387500) "
            "stationary=(0.600000,0.400000) simulated_state1=0.397433 "
            "poisson=(5.980833,5.988466) brownian_cov_error=0.006948 "
            "qv=(1.000731,0.007781) martingale=(-1.0,1.0) "
            "nonmarkov=(0.5,1.0)\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_eleven_with_process_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch11",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch11\n"
            "evidence=7/7\n"
            "oracle=passed markov_p5=(0.612500,0.387500) "
            "stationary=(0.600000,0.400000) simulated_state1=0.397433 "
            "poisson=(5.980833,5.988466) brownian_cov_error=0.006948 "
            "qv=(1.000731,0.007781) martingale=(-1.0,1.0) "
            "nonmarkov=(0.5,1.0)\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_twelve_notebook_reproduces_time_series_oracles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch12_time_series.py",
                "--output",
                "build/notebooks/upper/ch12_time_series.ipynb",
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
            "notebook=build/notebooks/upper/ch12_time_series.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed "
            "ar1=(-0.013814,2.764404,0.799071) "
            "forecast=(0.512000,2.049600) "
            "kalman=(0.555556,0.084615,0.152494,0.410431) "
            "spurious=(0.344011,0.000216) "
            "split_mse=(1.436939,2.226153)\n",
        )
        self.assertEqual(result.stderr, "")

    def test_accepts_chapter_twelve_with_time_series_oracles(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch12",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch12\n"
            "evidence=7/7\n"
            "oracle=passed ar1=(-0.013814,2.764404,0.799071) "
            "forecast=(0.512000,2.049600) "
            "kalman=(0.555556,0.084615,0.152494,0.410431) "
            "spurious=(0.344011,0.000216) "
            "split_mse=(1.436939,2.226153)\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()

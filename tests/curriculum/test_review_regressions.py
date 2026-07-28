from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class MathematicalReviewRegressionTests(unittest.TestCase):
    def chapter(self, number: int) -> str:
        return (ROOT / "tex" / "upper" / "chapters" / f"ch{number:02d}.tex").read_text(
            encoding="utf-8"
        )

    def test_reviewed_mathematical_statements_keep_their_required_conditions(self) -> None:
        stochastic = self.chapter(11)
        self.assertIn("X_0=i_0", stochastic)
        self.assertIn(r"\mathbb P(X_{n+1}=j\mid X_n=i)", stochastic)
        self.assertIn("条件事件概率为正", stochastic)
        self.assertIn(r"(nt-\lfloor nt\rfloor)\xi_{\lfloor nt\rfloor+1}", stochastic)
        self.assertIn(r"C[0,1]", stochastic)

        time_series = self.chapter(12)
        self.assertIn("弱白噪声不推出鞅差创新", time_series)
        self.assertIn(r"\mathbb E(\varepsilon_t\mid\mathcal F_{t-1})=0", time_series)

        conditioning = self.chapter(8)
        self.assertIn("$X$ 与 $Z$ 的状态空间均为标准 Borel 空间", conditioning)

        estimation = self.chapter(10)
        self.assertIn(r"\frac{X^TX}{n}\xrightarrow{p}Q", estimation)
        self.assertIn(r"\frac{X^T\varepsilon}{n}\xrightarrow{p}0", estimation)
        self.assertIn(r"Q\succ0", estimation)

        linear_algebra = self.chapter(6)
        self.assertIn(r"\lVert A-A_k\rVert_2=\sigma_{k+1}", linear_algebra)
        self.assertIn(
            r"\lVert A-A_k\rVert_F=\left(\sum_{i>k}\sigma_i^2\right)^{1/2}",
            linear_algebra,
        )

        notation = json.loads(
            (ROOT / "curriculum" / "notation.json").read_text(encoding="utf-8")
        )
        expectation = next(
            item for item in notation["symbols"] if item["symbol"] == r"\mathbb{E}[X]"
        )
        self.assertEqual(
            expectation["domain"],
            "real when X is integrable; extended real when defined",
        )


class PrerequisiteSingleSourceTests(unittest.TestCase):
    def test_chapter_prerequisites_are_generated_from_the_manifest(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "render_shared_registries.py"),
                "--check",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        upper_units = [
            unit for unit in manifest["units"] if unit["id"].startswith("upper.ch")
        ]
        self.assertEqual(manifest["prerequisite_semantics"], "direct")
        for unit in upper_units:
            number = int(unit["id"].removeprefix("upper.ch"))
            chapter = (
                ROOT / "tex" / "upper" / "chapters" / f"ch{number:02d}.tex"
            ).read_text(encoding="utf-8")
            generated = f"tex/generated/prerequisites/upper-ch{number:02d}.tex"
            self.assertIn(rf"\input{{{generated}}}", chapter)
            self.assertNotIn(r"\item 先修：", chapter)
            self.assertTrue((ROOT / generated).is_file())


class ReaderNavigationContractTests(unittest.TestCase):
    def test_manifest_drives_reader_routes_parts_and_course_map(self) -> None:
        manifest = json.loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        routes = manifest["reading_routes"]
        self.assertEqual({route["id"] for route in routes}, {"applied", "theory"})
        for route in routes:
            self.assertTrue(route["units"])
            self.assertTrue(all(unit.startswith("upper.ch") for unit in route["units"]))

        smoke = next(
            unit for unit in manifest["units"] if unit["id"] == "foundation.oracle-smoke"
        )
        self.assertTrue(smoke["internal"])

        for track in manifest["tracks"]:
            prerequisites = set(track["bridge_prerequisites"])
            self.assertIn("upper.ch16", prerequisites)
            self.assertIn("upper.ch17", prerequisites)

        upper_main = (ROOT / "tex" / "upper" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertEqual(upper_main.count(r"\part{"), 5)
        self.assertIn(r"\input{tex/generated/course-map}", upper_main)

        course_map = ROOT / "tex" / "generated" / "course-map.tex"
        self.assertTrue(course_map.is_file())
        rendered = course_map.read_text(encoding="utf-8")
        self.assertIn("应用主线", rendered)
        self.assertIn("理论增强线", rendered)
        self.assertNotIn("foundation.oracle-smoke", rendered)

        lower_main = (ROOT / "tex" / "lower" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\subtitle{下册：方向模型与研究项目}", lower_main)

        bridge = next(
            item
            for item in manifest["reader_bridges"]
            if item["id"] == "probability-minimal-measure"
        )
        self.assertEqual(bridge["for_routes"], ["applied"])
        self.assertEqual(bridge["before_unit"], "upper.ch07")
        bridge_source = ROOT / bridge["source"]
        self.assertTrue(bridge_source.is_file())
        bridge_input = rf"\input{{{bridge['source'].removesuffix('.tex')}}}"
        self.assertIn(bridge_input, upper_main)
        self.assertLess(upper_main.index(bridge_input), upper_main.index(r"\input{tex/upper/chapters/ch07}"))

        course_map_text = (ROOT / "tex" / "generated" / "course-map.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("概率所需最小测度论桥接", course_map_text)

    def test_editorial_environments_figures_and_concept_index_are_reader_visible(self) -> None:
        environments = ROOT / "tex" / "common" / "editorial-environments.tex"
        self.assertTrue(environments.is_file())
        definitions = environments.read_text(encoding="utf-8")
        for name in ("heuristic", "diagnostic", "implementationnote"):
            self.assertIn(rf"\newenvironment{{{name}}}", definitions)

        upper_main = (ROOT / "tex" / "upper" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\input{tex/common/editorial-environments}", upper_main)
        self.assertIn(r"\input{tex/generated/concept-index}", upper_main)

        expected_figures = {
            6: "projection-svd.tex",
            9: "convergence-map.tex",
            11: "filtration-timeline.tex",
            16: "purge-embargo.tex",
        }
        for chapter, filename in expected_figures.items():
            source = (
                ROOT / "tex" / "upper" / "chapters" / f"ch{chapter:02d}.tex"
            ).read_text(encoding="utf-8")
            self.assertIn(rf"\input{{tex/upper/figures/{filename}}}", source)
            self.assertTrue((ROOT / "tex" / "upper" / "figures" / filename).is_file())

        self.assertIn(r"\begin{diagnostic}", self.chapter_text(9))
        self.assertIn(r"\begin{implementationnote}", self.chapter_text(14))
        self.assertIn(r"\begin{heuristic}", self.chapter_text(16))

        concept_index = ROOT / "tex" / "generated" / "concept-index.tex"
        self.assertTrue(concept_index.is_file())
        rendered = concept_index.read_text(encoding="utf-8")
        self.assertIn("一致可积", rendered)
        self.assertIn("第 9 章", rendered)
        self.assertNotIn("foundation.oracle-smoke", rendered)

    @staticmethod
    def chapter_text(number: int) -> str:
        return (ROOT / "tex" / "upper" / "chapters" / f"ch{number:02d}.tex").read_text(
            encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()

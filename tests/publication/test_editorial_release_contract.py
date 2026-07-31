from __future__ import annotations

import re
import unittest
import json
from unittest import mock

import tools.render_shared_registries as registry_renderer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class EditorialReleaseContractTests(unittest.TestCase):
    def test_editorial_preamble_prevents_orphan_and_widow_lines(self) -> None:
        source = (ROOT / "tex/common/editorial-environments.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("\\clubpenalty=10000", source)
        self.assertIn("\\widowpenalty=10000", source)
        self.assertIn("\\displaywidowpenalty=10000", source)

    def test_upper_volume_is_citable_reproducible_and_honestly_described(self) -> None:
        upper_main = (ROOT / "tex" / "upper" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(r"\date{\today}", upper_main)
        self.assertIn(r"\date{\MFQReleaseDate}", upper_main)
        self.assertIn(r"\addbibresource{tex/common/references.bib}", upper_main)
        self.assertIn(r"\printbibliography[heading=none]", upper_main)

        references = (ROOT / "tex" / "common" / "references.bib").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(len(re.findall(r"(?m)^@", references)), 15)
        for number in range(1, 18):
            chapter = (
                ROOT / "tex" / "upper" / "chapters" / f"ch{number:02d}.tex"
            ).read_text(encoding="utf-8")
            self.assertIn(r"\cite{", chapter, f"chapter {number} has no references")

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        declared_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertIn(f"v{declared_version}", readme)
        self.assertIn("35 个正式学习单元", readme)
        self.assertIn("releases/latest", readme)
        self.assertNotIn("尚未开始正式章节写作", readme)

        for relative in (
            "ERRATA.md",
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            ".github/ISSUE_TEMPLATE/mathematical-error.yml",
            "VERSION",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.11,<3.14"', pyproject)

        build_driver = (ROOT / "tools" / "build_books.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"latexmk"', build_driver)
        self.assertIn('"-xelatex"', build_driver)
        self.assertNotIn("for _ in range(2)", build_driver)
        self.assertIn('environment.setdefault("SOURCE_DATE_EPOCH"', build_driver)

        release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("tags:", release_workflow)
        self.assertIn(
            "tools/build_release.py --vendored-template-only", release_workflow
        )
        self.assertIn("MFQ_RELEASE_TAG: ${{ github.ref_name }}", release_workflow)
        self.assertIn("math-for-quant-solutions.pdf", release_workflow)

        release_driver = (ROOT / "tools" / "build_release.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'run([sys.executable, "tools/build_books.py", "--volume", "all"])',
            release_driver,
        )

    def test_complete_solutions_are_published_as_one_shared_supplement(self) -> None:
        upper_main = (ROOT / "tex" / "upper" / "main.tex").read_text(
            encoding="utf-8"
        )
        for number in range(1, 18):
            self.assertNotIn(rf"\input{{tex/upper/solutions/ch{number:02d}}}", upper_main)

        solutions_main = ROOT / "tex" / "solutions-main.tex"
        self.assertTrue(solutions_main.is_file())
        solutions = solutions_main.read_text(encoding="utf-8")
        for number in range(1, 18):
            self.assertIn(rf"\input{{tex/upper/solutions/ch{number:02d}}}", solutions)
        manifest = __import__("json").loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        lower_solutions = {
            unit["evidence"]["solutions"]
            for unit in manifest["units"]
            if unit.get("volume") == "lower"
            and unit.get("published")
            and unit.get("state") == "accepted"
        }
        for path in lower_solutions:
            source = path.removesuffix(".tex")
            self.assertIn(rf"\input{{{source}}}", solutions)

        supplement = next(
            item for item in manifest["supplements"] if item["id"] == "solutions"
        )
        self.assertEqual(supplement["parent_volumes"], ["upper", "lower"])
        self.assertEqual(supplement["source"], "tex/solutions-main.tex")
        self.assertEqual(
            supplement["pdf"], "output/pdf/math-for-quant-solutions.pdf"
        )

        build_driver = (ROOT / "tools" / "build_books.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('selected_volume in item["parent_volumes"]', build_driver)

    def test_version_and_ci_are_shared_across_all_publications(self) -> None:
        for relative in ("tex/upper/main.tex", "tex/lower/main.tex", "tex/solutions-main.tex"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(r"\version{\MFQVersion}", source)
        build_driver = (ROOT / "tools" / "build_books.py").read_text(encoding="utf-8")
        self.assertIn("MFQVersion", build_driver)
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("tools/build_books.py --volume all", ci)
        self.assertIn('MFQ_SKIP_LATEX: "1"', ci)
        self.assertIn(
            "tools/check_learning_unit.py --manifest curriculum/manifest.json --track all",
            ci,
        )
        self.assertIn(
            "tools/check_learning_unit.py --manifest curriculum/manifest.json --volume all",
            ci,
        )

    def test_v03_metadata_and_lower_generated_navigation_are_release_ready(self) -> None:
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertEqual(re.search(r'(?m)^version = "([^"]+)"$', pyproject).group(1), version)
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertEqual(re.search(r"(?m)^version: (.+)$", citation).group(1), version)
        release_date = re.search(r"(?m)^date-released: (.+)$", citation).group(1)
        self.assertRegex(release_date, r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn(
            f"## {version} - {release_date}",
            (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        )
        release_workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn("name: ${{ github.ref_name }} — 完整双册与共享答案", release_workflow)
        self.assertIn("body_path: docs/releases/${{ github.ref_name }}.md", release_workflow)
        self.assertTrue((ROOT / "docs/releases" / f"v{version}.md").is_file())
        build_driver = (ROOT / "tools/build_books.py").read_text(encoding="utf-8")
        self.assertIn("release date is unavailable", build_driver)
        self.assertNotIn(f'return "{release_date}"', build_driver)

        lower_main = (ROOT / "tex/lower/main.tex").read_text(encoding="utf-8")
        self.assertNotIn(r"\setcounter{chapter}{0}", lower_main)
        self.assertLess(
            lower_main.index(r"\input{tex/lower/templates/capstone-evidence}"),
            lower_main.index(r"\mainmatter"),
        )
        self.assertIn(r"\input{tex/generated/lower-concept-index}", lower_main)

        manifest = json.loads((ROOT / "curriculum/manifest.json").read_text(encoding="utf-8"))
        for unit in manifest["units"]:
            if unit.get("volume") != "lower" or not unit.get("published"):
                continue
            chapter = (ROOT / unit["evidence"]["notation_and_assumptions"]).read_text(encoding="utf-8")
            generated = unit["id"].replace(".", "-")
            self.assertIn(
                rf"\input{{tex/generated/prerequisites/{generated}}}", chapter
            )

        config = json.loads((ROOT / "curriculum/visual-regression.json").read_text(encoding="utf-8"))
        selected = {
            publication["id"]: {page["id"] for page in publication["pages"]}
            for publication in config["publications"]
        }
        self.assertTrue({"formula", "table-or-code", "concept-index"} <= selected["lower"])
        self.assertTrue({"upper-solutions", "lower-solutions", "final-answer"} <= selected["solutions"])

    def test_generated_registry_inventory_rejects_obsolete_prerequisites(self) -> None:
        generated = ROOT / "build/test-generated-prerequisites"
        generated.mkdir(parents=True, exist_ok=True)
        expected_path = generated / "expected.tex"
        obsolete_path = generated / "obsolete.tex"
        expected_path.write_text("expected\n", encoding="utf-8")
        obsolete_path.write_text("obsolete\n", encoding="utf-8")
        with mock.patch.object(registry_renderer, "PREREQUISITE_OUTPUT", generated):
            observed = registry_renderer.obsolete_prerequisite_files(
                {expected_path: "expected\n"}
            )
        self.assertEqual(observed, [obsolete_path.resolve()])


if __name__ == "__main__":
    unittest.main()

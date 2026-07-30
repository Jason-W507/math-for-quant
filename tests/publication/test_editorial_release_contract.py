from __future__ import annotations

import re
import unittest
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
        self.assertIn("初稿已完成", readme)
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


if __name__ == "__main__":
    unittest.main()

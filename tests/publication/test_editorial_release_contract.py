from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class EditorialReleaseContractTests(unittest.TestCase):
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
        self.assertIn("math-for-quant-upper-solutions.pdf", release_workflow)

        release_driver = (ROOT / "tools" / "build_release.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'run([sys.executable, "tools/build_books.py", "--volume", "all"])',
            release_driver,
        )

    def test_complete_solutions_are_published_as_an_upper_supplement(self) -> None:
        upper_main = (ROOT / "tex" / "upper" / "main.tex").read_text(
            encoding="utf-8"
        )
        for number in range(1, 18):
            self.assertNotIn(rf"\input{{tex/upper/solutions/ch{number:02d}}}", upper_main)

        solutions_main = ROOT / "tex" / "upper" / "solutions-main.tex"
        self.assertTrue(solutions_main.is_file())
        solutions = solutions_main.read_text(encoding="utf-8")
        for number in range(1, 18):
            self.assertIn(rf"\input{{tex/upper/solutions/ch{number:02d}}}", solutions)

        manifest = __import__("json").loads(
            (ROOT / "curriculum" / "manifest.json").read_text(encoding="utf-8")
        )
        supplement = next(
            item for item in manifest["supplements"] if item["id"] == "upper-solutions"
        )
        self.assertEqual(supplement["parent_volume"], "upper")
        self.assertEqual(supplement["source"], "tex/upper/solutions-main.tex")
        self.assertEqual(
            supplement["pdf"], "output/pdf/math-for-quant-upper-solutions.pdf"
        )

        build_driver = (ROOT / "tools" / "build_books.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('item["parent_volume"] == selected_volume', build_driver)


if __name__ == "__main__":
    unittest.main()

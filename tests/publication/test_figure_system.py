from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FigureSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = json.loads(
            (ROOT / "figures" / "figure-specs.json").read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (ROOT / "figures" / "figure-manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_registers_the_complete_visual_system(self) -> None:
        records = self.manifest["figures"]
        ids = [record["id"] for record in records]
        self.assertGreaterEqual(self.manifest["figure_count"], 45)
        self.assertLessEqual(self.manifest["figure_count"], 60)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            len(self.specs["figures"]),
            len(records),
        )

    def test_cached_vector_assets_match_the_manifest(self) -> None:
        generated = [record for record in self.manifest["figures"] if record["cached_asset"]]
        self.assertEqual(len(self.specs["figures"]), len(generated))
        for record in generated:
            asset = ROOT / record["cached_asset"]
            wrapper = ROOT / record["tex_wrapper"]
            self.assertTrue(asset.is_file(), record["id"])
            self.assertTrue(wrapper.is_file(), record["id"])
            self.assertEqual(record["asset_sha256"], sha256(asset), record["id"])
            reader = PdfReader(asset)
            self.assertEqual(1, len(reader.pages), record["id"])
            if record["kind"] == "evidence":
                self.assertTrue(record["scene"], record["id"])
                self.assertTrue((ROOT / record["companion"]).is_file(), record["id"])

    def test_every_generated_wrapper_is_embedded_at_its_teaching_anchor(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for volume in ("upper", "lower")
            for path in (ROOT / "tex" / volume / "chapters").glob("*.tex")
        )
        for spec in self.specs["figures"]:
            chapter = (ROOT / spec["file"]).read_text(encoding="utf-8")
            include = f"\\input{{tex/figures/generated/{spec['id']}.tex}}"
            self.assertEqual(1, chapter.count(include), spec["id"])
            self.assertEqual(1, corpus.count(include), f"duplicate book-wide figure: {spec['id']}")
            self.assertIn(spec["anchor"], chapter, spec["id"])
            self.assertLess(chapter.index(spec["anchor"]), chapter.index(include), spec["id"])
            between = chapter[
                chapter.index(spec["anchor"]) + len(spec["anchor"]):chapter.index(include)
            ]
            prose_lines = []
            for line in between.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("\\label", "\\begin", "\\end", "\\input")):
                    continue
                prose_lines.append(stripped)
            self.assertTrue(prose_lines, f"figure precedes its teaching paragraph: {spec['id']}")

    def test_chapters_use_cached_assets_instead_of_inline_tikz(self) -> None:
        for chapter in (ROOT / "tex" / "lower" / "chapters").glob("*.tex"):
            self.assertNotIn("\\begin{tikzpicture}", chapter.read_text(encoding="utf-8"), chapter.name)
        for chapter in (ROOT / "tex" / "upper" / "chapters").glob("*.tex"):
            self.assertNotIn("\\begin{tikzpicture}", chapter.read_text(encoding="utf-8"), chapter.name)

    def test_ordinary_book_builds_do_not_execute_figure_generation(self) -> None:
        build_driver = (ROOT / "tools" / "build_books.py").read_text(encoding="utf-8")
        upper = (ROOT / "tex" / "upper" / "main.tex").read_text(encoding="utf-8")
        lower = (ROOT / "tex" / "lower" / "main.tex").read_text(encoding="utf-8")
        self.assertNotIn("build_figures", build_driver)
        self.assertNotIn("write18", upper + lower)
        self.assertNotIn("shell-escape", upper + lower)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

import fitz

from tools.check_pdf_visual_regression import (
    hamming,
    observed_hashes,
    validate_pdf_structure,
)


ROOT = Path(__file__).resolve().parents[2]


class PdfVisualRegressionTests(unittest.TestCase):
    def test_key_pages_are_selected_by_number_or_unique_bookmark(self) -> None:
        pdf = ROOT / "build" / "test-visual" / "sample.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open() as document:
            first = document.new_page()
            first.insert_text((72, 72), "Cover")
            second = document.new_page()
            second.insert_text((72, 72), "Unique research marker")
            document.set_toc([[1, "Research section", 2]])
            document.save(pdf)
        config = {
            "hash_size": 8,
            "publications": [
                {
                    "id": "sample",
                    "pdf": pdf.relative_to(ROOT).as_posix(),
                    "pages": [
                        {"id": "cover", "page": 1},
                        {"id": "research", "bookmark": "Research section"},
                    ],
                }
            ],
        }
        observed = observed_hashes(config)
        self.assertEqual(set(observed), {"sample:cover", "sample:research"})
        self.assertNotEqual(observed["sample:cover"], observed["sample:research"])
        self.assertEqual(hamming(observed["sample:cover"], observed["sample:cover"]), 0)

    def test_structure_gate_rejects_missing_required_metadata(self) -> None:
        pdf = ROOT / "build" / "test-visual" / "missing-metadata.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open() as document:
            document.new_page()
            document.save(pdf)
        with fitz.open(pdf) as document:
            with self.assertRaisesRegex(ValueError, "metadata title"):
                validate_pdf_structure(
                    document,
                    {
                        "id": "sample",
                        "metadata": {"title": "Expected", "author": "Author"},
                    },
                )

    def test_structure_gate_rejects_nonmonotone_bookmark_destinations(self) -> None:
        pdf = ROOT / "build" / "test-visual" / "nonmonotone-toc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open() as document:
            document.new_page()
            document.new_page()
            document.set_metadata({"title": "Expected", "author": "Author"})
            document.set_toc([[1, "Later", 2], [1, "Earlier", 1]])
            document.save(pdf)
        with fitz.open(pdf) as document:
            with self.assertRaisesRegex(ValueError, "bookmark destinations are not monotone"):
                validate_pdf_structure(
                    document,
                    {
                        "id": "sample",
                        "metadata": {"title": "Expected", "author": "Author"},
                    },
                )


if __name__ == "__main__":
    unittest.main()

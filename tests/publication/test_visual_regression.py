from __future__ import annotations

import unittest
from pathlib import Path

import fitz

from tools.check_pdf_visual_regression import hamming, observed_hashes


ROOT = Path(__file__).resolve().parents[2]


class PdfVisualRegressionTests(unittest.TestCase):
    def test_key_pages_are_selected_by_number_or_unique_text(self) -> None:
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


if __name__ == "__main__":
    unittest.main()

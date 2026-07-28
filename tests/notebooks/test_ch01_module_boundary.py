from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ChapterOneModuleBoundaryTests(unittest.TestCase):
    def test_notebook_is_import_safe_and_delegates_to_an_importable_module(self) -> None:
        module = ROOT / "src" / "math_for_quant" / "ch01" / "convex_bound.py"
        self.assertTrue(module.is_file())

        notebook = ROOT / "notebooks" / "upper" / "ch01_convex_bound.py"
        source = notebook.read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("from math_for_quant.ch01.convex_bound import", source)
        namespace = runpy.run_path(str(notebook), run_name="notebook_import_test")
        self.assertIn("main", namespace)

    def test_oracle_is_bound_to_a_separate_fixture_by_sha256(self) -> None:
        oracle_path = ROOT / "evidence" / "ch01" / "oracle.json"
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        fixture = ROOT / oracle["fixture"]["path"]
        self.assertTrue(fixture.is_file())
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        self.assertEqual(digest, oracle["fixture"]["sha256"])
        for input_field in ("returns", "weights", "counterexample_weights"):
            self.assertNotIn(input_field, oracle)

        spec = importlib.util.spec_from_file_location("convex_bound", module_path())
        self.assertIsNotNone(spec)
        loaded = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(loaded)
        result = loaded.run_contract(fixture, oracle_path)
        self.assertEqual(result.weighted_return, 0.013)


def module_path() -> Path:
    return ROOT / "src" / "math_for_quant" / "ch01" / "convex_bound.py"


if __name__ == "__main__":
    unittest.main()

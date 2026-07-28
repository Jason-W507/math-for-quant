from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[2]


from tests.support.learning_unit_case import LearningUnitCase


class CapstoneTests(LearningUnitCase):
    def test_chapter_seventeen_notebook_reproduces_clean_research_package(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "build_notebook.py"),
                "--source",
                "notebooks/upper/ch17_research_audit.py",
                "--output",
                "build/notebooks/upper/ch17_research_audit.ipynb",
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
            "notebook=build/notebooks/upper/ch17_research_audit.ipynb\n"
            "roundtrip=passed cells=2\n"
            "execution=passed oracle=passed audit=passed package=upper-capstone-v1 "
            "data=research_rows.csv rows=3 timeline=passed split=passed "
            "multiplicity=passed "
            "performance=(gross=0.040000,cost=0.006000,net=0.034000) "
            "numeric=passed license=CC0-1.0 licenses=4 limitations=6\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_seventeen_package_runs_from_a_clean_copy(self) -> None:
        result = self.run_chapter_seventeen_package_fixture(
            "clean",
            lambda _root, _oracle: None,
            use_declared_command=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "oracle=passed audit=passed package=upper-capstone-v1 "
            "data=research_rows.csv rows=3 timeline=passed split=passed "
            "multiplicity=passed "
            "performance=(gross=0.040000,cost=0.006000,net=0.034000) "
            "numeric=passed license=CC0-1.0 licenses=4 limitations=6\n",
        )
        self.assertEqual(result.stderr, "")

    def test_chapter_seventeen_rejects_leakage_after_integrity_update(self) -> None:
        def introduce_leakage(root: Path, oracle: dict[str, Any]) -> None:
            data_path = root / oracle["data_path"]
            text = data_path.read_text(encoding="utf-8").replace(
                "2024-07-01T08:00:00+08:00",
                "2024-07-01T09:31:00+08:00",
                1,
            )
            data_path.write_text(text, encoding="utf-8")
            oracle["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()

        result = self.run_chapter_seventeen_package_fixture(
            "leakage",
            introduce_leakage,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "timeline gate failed: row 1 was unavailable at decision\n",
        )

    def test_chapter_seventeen_rejects_a_changed_limitation_report(self) -> None:
        def change_limitations(root: Path, oracle: dict[str, Any]) -> None:
            limitations = root / oracle["limitations_path"]
            text = limitations.read_text(encoding="utf-8").replace(
                "cannot establish generalization",
                "proves broad generalization",
                1,
            )
            limitations.write_text(text, encoding="utf-8")

        result = self.run_chapter_seventeen_package_fixture(
            "changed-limitations",
            change_limitations,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "report gate failed: limitation report checksum mismatch\n",
        )

    def test_chapter_seventeen_rejects_a_missing_code_license(self) -> None:
        def remove_code_license(root: Path, _oracle: dict[str, Any]) -> None:
            (root / "LICENSE").write_text(
                "No license declaration.\n",
                encoding="utf-8",
            )

        result = self.run_chapter_seventeen_package_fixture(
            "missing-code-license",
            remove_code_license,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "license gate failed: code license marker is missing\n",
        )

    def test_chapter_seventeen_rejects_a_vacuous_code_license_entry(self) -> None:
        def empty_code_protocol(root: Path, oracle: dict[str, Any]) -> None:
            manifest_path = root / oracle["license_manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            code = next(
                asset for asset in manifest["assets"] if asset["id"] == "code"
            )
            code["paths"] = []
            code["required_marker"] = ""
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            oracle["license_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()

        result = self.run_chapter_seventeen_package_fixture(
            "vacuous-code-license",
            empty_code_protocol,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "license gate failed: code asset paths are missing\n",
        )

    def test_chapter_seventeen_rejects_a_license_path_outside_the_package(
        self,
    ) -> None:
        def escape_to_original_checkout(root: Path, oracle: dict[str, Any]) -> None:
            (root / "LICENSE").write_text(
                "No license declaration.\n",
                encoding="utf-8",
            )
            manifest_path = root / oracle["license_manifest_path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            code = next(
                asset for asset in manifest["assets"] if asset["id"] == "code"
            )
            code["license_file"] = str((ROOT / "LICENSE").resolve())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            oracle["license_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()

        result = self.run_chapter_seventeen_package_fixture(
            "escaped-code-license",
            escape_to_original_checkout,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            result.stderr,
            "license gate failed: code license file escapes package root\n",
        )

    def test_accepts_chapter_seventeen_with_capstone_audit(self) -> None:
        result = self.run_contract(
            "curriculum/manifest.json",
            "--unit",
            "upper.ch17",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "unit=upper.ch17\n"
            "evidence=7/7\n"
            "oracle=passed audit=passed package=upper-capstone-v1 "
            "data=research_rows.csv rows=3 timeline=passed split=passed "
            "multiplicity=passed "
            "performance=(gross=0.040000,cost=0.006000,net=0.034000) "
            "numeric=passed license=CC0-1.0 licenses=4 limitations=6\n"
            "learning-unit contract passed\n",
        )
        self.assertEqual(result.stderr, "")


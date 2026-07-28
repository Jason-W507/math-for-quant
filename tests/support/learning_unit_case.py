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


class LearningUnitCase(unittest.TestCase):
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

    def run_oracle_fixture(
        self,
        fixture_name: str,
        oracle_path: str,
        script_path: str,
        mutate: Callable[[dict[str, Any]], None],
    ) -> subprocess.CompletedProcess[str]:
        oracle = json.loads(
            (ROOT / oracle_path).read_text(encoding="utf-8")
        )
        fixture_contract = oracle["fixture"]
        fixture_document = json.loads(
            (ROOT / fixture_contract["path"]).read_text(encoding="utf-8")
        )
        fixture_fields = set(fixture_document)
        document = {**fixture_document, **oracle}
        mutate(document)
        fixture = ROOT / "build" / "test-fixtures" / fixture_name
        input_fixture = fixture.with_name(f"{fixture.stem}-input.json")
        fixture.parent.mkdir(parents=True, exist_ok=True)
        mutated_input = {
            key: document[key] for key in fixture_fields if key in document
        }
        input_bytes = json.dumps(mutated_input).encode("utf-8")
        input_fixture.write_bytes(input_bytes)
        mutated_oracle = {
            key: value for key, value in document.items() if key not in fixture_fields
        }
        mutated_oracle["fixture"] = {
            "path": str(input_fixture.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        }
        fixture.write_text(json.dumps(mutated_oracle), encoding="utf-8")
        try:
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / script_path),
                    str(fixture),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            fixture.unlink(missing_ok=True)
            input_fixture.unlink(missing_ok=True)

    def run_chapter_seventeen_package_fixture(
        self,
        fixture_name: str,
        mutate: Callable[[Path, dict[str, Any]], None],
        *,
        use_declared_command: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        fixture_root = ROOT / "build" / "test-packages" / fixture_name
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        oracle_contract = json.loads(
            (ROOT / "evidence" / "ch17" / "oracle.json").read_text(
                encoding="utf-8"
            )
        )
        canonical_fixture = json.loads(
            (ROOT / oracle_contract["fixture"]["path"]).read_text(encoding="utf-8")
        )
        for relative in canonical_fixture["package_files"]:
            source = ROOT / relative
            target = fixture_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        oracle_path = fixture_root / "evidence" / "ch17" / "oracle.json"
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        input_path = fixture_root / oracle["fixture"]["path"]
        input_document = json.loads(input_path.read_text(encoding="utf-8"))
        input_fields = set(input_document)
        document = {**input_document, **oracle}
        mutate(fixture_root, document)
        input_document = {
            key: document[key] for key in input_fields if key in document
        }
        input_bytes = json.dumps(input_document, indent=2).encode("utf-8")
        input_path.write_bytes(input_bytes)
        oracle = {
            key: value for key, value in document.items() if key not in input_fields
        }
        oracle["fixture"] = {
            "path": str(input_path.relative_to(fixture_root)).replace("\\", "/"),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        }
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        try:
            if use_declared_command:
                command = list(document["command"])
                command[0] = shutil.which(command[0]) or command[0]
            else:
                command = [
                    sys.executable,
                    str(
                        fixture_root
                        / "notebooks"
                        / "upper"
                        / "ch17_research_audit.py"
                    ),
                    str(oracle_path),
                    str(fixture_root),
                ]
            return subprocess.run(
                command,
                cwd=fixture_root,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            shutil.rmtree(fixture_root, ignore_errors=True)


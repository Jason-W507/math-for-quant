from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [sys.executable, "tools/check_learning_unit.py", "--manifest", "curriculum/manifest.json", "--track", "stat-arb"],
        [sys.executable, "tools/build_books.py", "--volume", "lower"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    print("stat-arb-route=passed publications=lower+shared-solutions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

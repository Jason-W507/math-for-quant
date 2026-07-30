from __future__ import annotations

try:
    from tools.validate_route import subprocess, validate_route
except ModuleNotFoundError:  # Direct script execution adds tools/, not repo root.
    from validate_route import subprocess, validate_route


def main() -> int:
    return validate_route("multifactor")


if __name__ == "__main__":
    raise SystemExit(main())

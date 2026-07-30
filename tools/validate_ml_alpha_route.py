from __future__ import annotations

try:
    from tools.validate_route import subprocess, validate_route
except ModuleNotFoundError:
    from validate_route import subprocess, validate_route


def main() -> int:
    return validate_route("ml-alpha")


if __name__ == "__main__":
    raise SystemExit(main())

"""Check that the vendored TRIPSO package is installed and importable."""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    """Import TRIPSO and report the active interpreter and module path."""
    print(f"Python executable: {sys.executable}")

    try:
        tripso = importlib.import_module("tripso")
    except ModuleNotFoundError as exc:
        if exc.name != "tripso":
            raise
        print(
            "TRIPSO could not be imported. Install the vendored package with:\n"
            "  python -m pip install -e ./tripso_code/tripso",
            file=sys.stderr,
        )
        return 1

    print(f"TRIPSO module path: {tripso.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

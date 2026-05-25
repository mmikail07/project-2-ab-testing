"""Regenerate processed data and execute notebooks 01 through 06 in order.
Used for the 'clone and run from scratch' reproducer.
"""
from __future__ import annotations

import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def regenerate_data() -> int:
    print(">>> regenerating synthetic experiment data")
    return subprocess.call(
        [sys.executable, "-m", "src.simulate"],
        cwd=ROOT,
    )


def execute_notebooks() -> int:
    nb_dir = ROOT / "notebooks"
    notebooks = sorted(nb_dir.glob("0*.ipynb"))
    if not notebooks:
        print("no notebooks found", file=sys.stderr)
        return 1
    for nb in notebooks:
        print(f">>> executing {nb.name}")
        rc = subprocess.call(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=300",
                str(nb),
            ],
            cwd=ROOT,
        )
        if rc != 0:
            print(f"FAILED: {nb.name}", file=sys.stderr)
            return rc
    return 0


def main() -> int:
    rc = regenerate_data()
    if rc != 0:
        return rc
    return execute_notebooks()


if __name__ == "__main__":
    raise SystemExit(main())

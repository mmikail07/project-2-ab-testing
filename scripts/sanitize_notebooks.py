"""Strip local-path leaks from notebook source cells before committing.

Project 1's notebooks accidentally printed absolute Windows paths in diagnostic
cells (`c:/Users/Lenovo/...`). This helper rewrites known offenders to use
relative-only forms. Run after editing notebooks before `git add`.

Currently configured to replace:
- bare `config.PROCESSED_DIR)` calls inside print statements with `config.PROCESSED_DIR.name`
- bare `config.PROJECT_ROOT)` calls with `config.PROJECT_ROOT.name`
"""
from __future__ import annotations

import pathlib
import sys

import nbformat

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"

REPLACEMENTS = [
    ("config.PROCESSED_DIR)", "config.PROCESSED_DIR.name)"),
    ("config.PROJECT_ROOT)", "config.PROJECT_ROOT.name)"),
    ("config.RAW_DIR)", "config.RAW_DIR.name)"),
]


def sanitize_notebook(path: pathlib.Path) -> int:
    nb = nbformat.read(path, as_version=4)
    edits = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        original = cell.source
        new = original
        for old, repl in REPLACEMENTS:
            if old in new and repl not in new:
                new = new.replace(old, repl)
        if new != original:
            cell.source = new
            edits += 1
    if edits:
        nbformat.write(nb, path)
    return edits


def main() -> int:
    notebooks = sorted(NB_DIR.glob("0*.ipynb"))
    if not notebooks:
        print(f"no notebooks found under {NB_DIR}", file=sys.stderr)
        return 1
    for nb in notebooks:
        n = sanitize_notebook(nb)
        print(f"{nb.name}: {n} cell(s) sanitized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

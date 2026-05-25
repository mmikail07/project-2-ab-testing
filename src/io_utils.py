"""Small I/O helpers so notebooks and scripts share one parquet / JSON layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_manifest(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_events(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_events(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _default_serializer(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

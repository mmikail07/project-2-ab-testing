"""Project-wide paths, seeds, and constants. Resolved relative to this file so
imports work whether invoked from repo root, a notebook, or a test runner."""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR") or (DATA_DIR / "processed"))
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EXPERIMENT_EVENTS_PARQUET = PROCESSED_DIR / "experiment_events.parquet"
EXPERIMENT_SUMMARY_CSV = PROCESSED_DIR / "experiment_summary.csv"
GROUND_TRUTH_JSON = PROCESSED_DIR / "ground_truth.json"
FUNNEL_BASELINE_JSON = EXTERNAL_DIR / "funnel_baseline.json"

SEED = int(os.getenv("EXPERIMENT_SEED", "42"))

DEFAULT_N_USERS = 50_000
DEFAULT_TRUE_LIFT_PP = 2.5
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_AA_SIMS = 10_000


def load_funnel_baseline() -> dict:
    with FUNNEL_BASELINE_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def baseline_completion_rate() -> float:
    """Marginal P(completion) under control, derived from the conditional chain."""
    baseline = load_funnel_baseline()
    rate = 1.0
    for stage in baseline["stages"]:
        rate *= stage["conditional_rate"]
    return rate

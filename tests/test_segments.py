"""Tests for src.segments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.segments import adjust_pvalues, run_segments


def _toy_events(n=4_000, seed=0):
    rng = np.random.default_rng(seed)
    arm = np.where(rng.random(n) < 0.5, "control", "treatment")
    is_returning = rng.random(n) < 0.4
    is_mobile = rng.random(n) < 0.7
    base = 0.10
    p = np.where(arm == "treatment", base + 0.02, base)
    completion = rng.random(n) < p
    return pd.DataFrame({
        "arm": arm,
        "is_returning": is_returning,
        "is_mobile": is_mobile,
        "completion": completion,
    })


def test_run_segments_returns_one_row_per_segment_value():
    df = _toy_events()
    result = run_segments(df, segment_cols=["is_returning", "is_mobile"])
    assert len(result) == 4
    assert {"is_returning", "is_mobile"}.issubset(set(result["segment_column"].unique()))


def test_run_segments_lift_signs_positive_when_lift_real():
    df = _toy_events(n=20_000)
    result = run_segments(df, segment_cols=["is_returning"])
    assert (result["lift_pp"] > 0).all()


def test_adjust_pvalues_adds_corrected_columns():
    df = _toy_events()
    result = run_segments(df, segment_cols=["is_returning", "is_mobile"])
    out = adjust_pvalues(result)
    assert "pvalue_bonferroni" in out.columns
    assert "pvalue_bh" in out.columns
    assert "reject_bonferroni" in out.columns
    assert "reject_bh" in out.columns


def test_bonferroni_more_conservative_than_bh():
    """Bonferroni-adjusted p-values >= BH-adjusted for any family."""
    df = _toy_events()
    result = run_segments(df, segment_cols=["is_returning", "is_mobile"])
    out = adjust_pvalues(result)
    assert (out["pvalue_bonferroni"] >= out["pvalue_bh"] - 1e-12).all()


def test_run_segments_raises_on_missing_column():
    df = _toy_events()
    import pytest
    with pytest.raises(KeyError):
        run_segments(df, segment_cols=["nonexistent_col"])

"""Segment-level analysis with multiple-comparison correction.

Slices the experiment by one or more boolean segment columns, runs a two-
proportion z-test inside each slice, then applies both Bonferroni and
Benjamini-Hochberg corrections to the family of p-values.

The notebook narrative is: Bonferroni controls family-wise error (the chance
of at least one false positive) — conservative, the right choice when one
false positive is costly. Benjamini-Hochberg controls false-discovery rate
(the expected share of false positives among rejections) — permissive, the
right choice for exploration. Showing both together makes the trade-off
explicit and is the canonical interview signal for senior analyst roles.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from statsmodels.stats.multitest import multipletests

from .stats import two_proportion_ztest


def run_segments(
    events: pd.DataFrame,
    segment_cols: Iterable[str],
    metric: str = "completion",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per (segment_col, segment_value, arm-tested-against-control)
    with z-test result and raw p-value. Apply multiple-comparison correction
    on the resulting frame with adjust_pvalues()."""
    rows: list[dict] = []
    for col in segment_cols:
        if col not in events.columns:
            raise KeyError(f"segment column {col!r} not in events")
        for value in (False, True):
            slice_df = events[events[col] == value]
            control = slice_df[slice_df["arm"] == "control"]
            treatment = slice_df[slice_df["arm"] == "treatment"]
            if len(control) == 0 or len(treatment) == 0:
                continue
            result = two_proportion_ztest(
                successes_c=int(control[metric].sum()),
                n_c=int(len(control)),
                successes_t=int(treatment[metric].sum()),
                n_t=int(len(treatment)),
                alpha=alpha,
            )
            rows.append({
                "segment_column": col,
                "segment_value": value,
                "n_control": result.n_control,
                "n_treatment": result.n_treatment,
                "rate_control": control[metric].mean(),
                "rate_treatment": treatment[metric].mean(),
                "lift_pp": result.point_estimate * 100,
                "ci_low_pp": result.ci_low * 100,
                "ci_high_pp": result.ci_high * 100,
                "pvalue": result.pvalue,
            })
    return pd.DataFrame(rows)


def adjust_pvalues(segments_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add bonferroni and benjamini-hochberg corrected p-values plus reject
    flags. Operates on the output of run_segments()."""
    if segments_df.empty:
        return segments_df.assign(
            pvalue_bonferroni=[], pvalue_bh=[], reject_bonferroni=[], reject_bh=[],
        )
    pvalues = segments_df["pvalue"].to_numpy()
    bonf_reject, bonf_p, _, _ = multipletests(pvalues, alpha=alpha, method="bonferroni")
    bh_reject, bh_p, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    out = segments_df.copy()
    out["pvalue_bonferroni"] = bonf_p
    out["pvalue_bh"] = bh_p
    out["reject_bonferroni"] = bonf_reject
    out["reject_bh"] = bh_reject
    return out

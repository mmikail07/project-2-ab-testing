"""Tests for src.stats. Cross-checked against statsmodels oracle implementations."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.proportion import proportions_ztest

from src.stats import (
    cohen_h,
    proportion_ci,
    srm_chi_square,
    two_proportion_ztest,
    welch_ttest,
)


def test_two_proportion_ztest_matches_statsmodels_pvalue():
    """statsmodels.proportions_ztest computes (p_first - p_second); we compute
    (p_treatment - p_control). Compare |z| and p-values (two-sided p is sign-invariant)."""
    successes = np.array([2400, 2550])
    nobs = np.array([50_000, 50_000])
    oracle_z, oracle_p = proportions_ztest(successes, nobs)
    result = two_proportion_ztest(2400, 50_000, 2550, 50_000)
    assert abs(abs(result.statistic) - abs(oracle_z)) < 1e-6
    assert abs(result.pvalue - oracle_p) < 1e-6


def test_two_proportion_ztest_point_estimate_sign():
    """Positive when treatment > control."""
    result = two_proportion_ztest(50, 1000, 80, 1000)
    assert result.point_estimate > 0
    assert result.ci_low < result.point_estimate < result.ci_high


def test_two_proportion_ztest_ci_excludes_zero_when_significant():
    result = two_proportion_ztest(50, 1000, 150, 1000)
    assert result.pvalue < 0.001
    assert result.ci_low > 0


def test_two_proportion_ztest_ci_includes_zero_when_null():
    """For balanced data with same counts, CI should bracket zero."""
    result = two_proportion_ztest(100, 1000, 100, 1000)
    assert result.pvalue > 0.5
    assert result.ci_low < 0 < result.ci_high


def test_welch_t_matches_scipy_for_unequal_variance():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 10, size=500)
    b = rng.normal(102, 14, size=500)
    oracle = scipy_stats.ttest_ind(b, a, equal_var=False)
    result = welch_ttest(a, b)
    assert abs(result.statistic - oracle.statistic) < 1e-6
    assert abs(result.pvalue - oracle.pvalue) < 1e-6


def test_welch_drops_nans():
    a = np.array([10, 12, np.nan, 14, 11])
    b = np.array([12, 14, 13, np.nan, 15])
    result = welch_ttest(a, b)
    assert result.n_control == 4
    assert result.n_treatment == 4


def test_proportion_ci_wilson_within_bounds():
    low, high = proportion_ci(50, 1000)
    assert 0 < low < 0.05 < high < 0.1


def test_proportion_ci_handles_zero():
    low, high = proportion_ci(0, 100)
    assert low >= 0
    assert high > 0


def test_cohen_h_matches_textbook_formula():
    """Cohen's h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))."""
    p1, p2 = 0.5, 0.4
    expected = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    assert abs(cohen_h(p1, p2) - expected) < 1e-12


def test_srm_chi_square_passes_on_balanced_split():
    """50/50 should produce a non-significant chi-square."""
    chi2, p = srm_chi_square([25_000, 25_000])
    assert p > 0.05


def test_srm_chi_square_flags_unbalanced_split():
    """A 55/45 split on 50k users is severe enough to trigger SRM alarm."""
    chi2, p = srm_chi_square([27_500, 22_500])
    assert p < 0.001

"""Tests for src.preflight. The A/A FPR assertion is the single strongest
guarantee that the statistical machinery is correct: under the null (no real
effect), the empirical Type I rate must land within tolerance of alpha."""
from __future__ import annotations

import numpy as np

from src.preflight import check_srm, run_aa


def test_srm_pass_on_balanced_assignment():
    out = check_srm({"control": 25_000, "treatment": 25_000})
    assert out["verdict"] == "pass"
    assert out["pvalue"] > 0.05
    assert out["shares"] == [0.5, 0.5]


def test_srm_alarm_on_severe_imbalance():
    out = check_srm({"control": 27_500, "treatment": 22_500})
    assert out["verdict"] == "alarm"
    assert out["pvalue"] < 0.001


def test_srm_uses_expected_ratio_when_provided():
    """A 60/40 split is correct if that's what we asked for. Should pass."""
    out = check_srm(
        {"control": 30_000, "treatment": 20_000},
        expected_ratio={"control": 0.6, "treatment": 0.4},
    )
    assert out["verdict"] == "pass"


def test_aa_empirical_fpr_close_to_alpha():
    """The headline guarantee: 10k A/A simulations at baseline 5% should yield
    an empirical Type I rate in [0.04, 0.06]. If this fails the stats machinery
    is wrong, full stop."""
    result = run_aa(n_per_arm=10_000, baseline=0.05, n_sims=10_000, seed=1)
    assert 0.04 <= result.empirical_fpr <= 0.06, (
        f"A/A FPR drifted to {result.empirical_fpr:.4f} (expected in [0.04, 0.06])"
    )


def test_aa_pvalue_distribution_approximately_uniform():
    """Under the null, p-values should be uniform on [0, 1]. Check via KS
    test against uniform with a generous tolerance."""
    from scipy.stats import kstest
    result = run_aa(n_per_arm=5_000, baseline=0.05, n_sims=5_000, seed=2)
    stat, p = kstest(result.pvalues, "uniform")
    assert p > 0.01, f"p-value distribution diverges from uniform (KS p={p:.4f})"


def test_aa_returns_correct_array_shape():
    result = run_aa(n_per_arm=2_000, baseline=0.05, n_sims=1_500, seed=3)
    assert result.pvalues.shape == (1_500,)
    assert (0 <= result.pvalues).all() and (result.pvalues <= 1).all()

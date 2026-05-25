"""Pre-experiment gates: Sample Ratio Mismatch and A/A validation.

SRM check delegates to src.stats.srm_chi_square. A/A simulation is fully
vectorized: a single numpy.binomial call generates all n_sims experiments at
once, z-statistics are evaluated as a vector operation. Target runtime is
under 2 seconds for n_sims=10_000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .stats import srm_chi_square


@dataclass
class AARunResult:
    n_sims: int
    n_per_arm: int
    baseline: float
    alpha: float
    empirical_fpr: float
    pvalues: np.ndarray
    seed: int


def check_srm(arm_counts: dict[str, int], expected_ratio: dict[str, float] | None = None) -> dict:
    """SRM chi-square. Returns dict with chi2, pvalue, arm shares, and a verdict
    string ('pass' if p >= 0.001, else 'alarm')."""
    arms = list(arm_counts.keys())
    counts = [arm_counts[a] for a in arms]
    if expected_ratio is None:
        ratios = None
    else:
        ratios = [expected_ratio[a] for a in arms]
    chi2, pvalue = srm_chi_square(counts, ratios)
    total = sum(counts)
    return {
        "arms": arms,
        "counts": counts,
        "shares": [c / total for c in counts],
        "chi2": chi2,
        "pvalue": pvalue,
        "verdict": "pass" if pvalue >= 0.001 else "alarm",
    }


def run_aa(
    n_per_arm: int,
    baseline: float,
    n_sims: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> AARunResult:
    """Run n_sims A/A experiments at the given sample size and baseline.

    Both arms draw from Bin(n_per_arm, baseline). The two-proportion z-test
    is evaluated vectorized across all sims. Empirical Type I error is the
    share of p-values below alpha.
    """
    rng = np.random.default_rng(seed)
    successes_c = rng.binomial(n_per_arm, baseline, size=n_sims).astype(np.int64)
    successes_t = rng.binomial(n_per_arm, baseline, size=n_sims).astype(np.int64)

    p_c = successes_c / n_per_arm
    p_t = successes_t / n_per_arm
    p_pool = (successes_c + successes_t) / (2 * n_per_arm)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per_arm)
    se = np.where(se == 0, 1e-12, se)
    z = (p_t - p_c) / se
    pvalues = 2 * (1 - norm.cdf(np.abs(z)))
    empirical_fpr = float((pvalues < alpha).mean())

    return AARunResult(
        n_sims=int(n_sims),
        n_per_arm=int(n_per_arm),
        baseline=float(baseline),
        alpha=float(alpha),
        empirical_fpr=empirical_fpr,
        pvalues=pvalues,
        seed=int(seed),
    )

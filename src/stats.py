"""Two-sample tests for the experiment analysis. Hand-rolled so notebooks can
show the formula plainly; cross-checked against statsmodels in tests/test_stats.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    statistic: float
    pvalue: float
    n_control: int
    n_treatment: int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def two_proportion_ztest(
    successes_c: int,
    n_c: int,
    successes_t: int,
    n_t: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> TestResult:
    """Pooled two-proportion z-test for binary outcomes.

    Point estimate: absolute difference p_t - p_c (in proportion units, not pp).
    CI: normal-approximation Wald interval on the difference.
    """
    if n_c <= 0 or n_t <= 0:
        raise ValueError("sample sizes must be positive")
    p_c = successes_c / n_c
    p_t = successes_t / n_t
    diff = p_t - p_c

    p_pool = (successes_c + successes_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se_pool == 0:
        z = 0.0
    else:
        z = diff / se_pool

    if alternative == "two-sided":
        pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
        z_crit = stats.norm.ppf(1 - alpha / 2)
    elif alternative == "larger":
        pvalue = 1 - stats.norm.cdf(z)
        z_crit = stats.norm.ppf(1 - alpha)
    else:
        pvalue = stats.norm.cdf(z)
        z_crit = stats.norm.ppf(1 - alpha)

    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    margin = z_crit * se_unpooled

    return TestResult(
        name="two-proportion z-test",
        point_estimate=float(diff),
        ci_low=float(diff - margin),
        ci_high=float(diff + margin),
        statistic=float(z),
        pvalue=float(pvalue),
        n_control=int(n_c),
        n_treatment=int(n_t),
    )


def welch_ttest(
    values_c: np.ndarray,
    values_t: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test for continuous metrics with unequal variance."""
    vc = np.asarray(values_c, dtype=float)
    vt = np.asarray(values_t, dtype=float)
    vc = vc[~np.isnan(vc)]
    vt = vt[~np.isnan(vt)]
    if len(vc) < 2 or len(vt) < 2:
        raise ValueError("need at least 2 observations per arm for Welch's t-test")

    mean_c = float(vc.mean())
    mean_t = float(vt.mean())
    diff = mean_t - mean_c
    var_c = float(vc.var(ddof=1))
    var_t = float(vt.var(ddof=1))
    se = float(np.sqrt(var_c / len(vc) + var_t / len(vt)))

    if se == 0:
        t = 0.0
        df = float(len(vc) + len(vt) - 2)
    else:
        t = diff / se
        df = (var_c / len(vc) + var_t / len(vt)) ** 2 / (
            (var_c / len(vc)) ** 2 / (len(vc) - 1)
            + (var_t / len(vt)) ** 2 / (len(vt) - 1)
        )

    pvalue = float(2 * (1 - stats.t.cdf(abs(t), df=df)))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    margin = t_crit * se

    return TestResult(
        name="welch t-test",
        point_estimate=float(diff),
        ci_low=float(diff - margin),
        ci_high=float(diff + margin),
        statistic=float(t),
        pvalue=pvalue,
        n_control=int(len(vc)),
        n_treatment=int(len(vt)),
    )


def proportion_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion. More reliable than Wald
    near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - margin), float(centre + margin))


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def srm_chi_square(observed_counts: list[int], expected_ratio: list[float] | None = None) -> tuple[float, float]:
    """Sample Ratio Mismatch chi-square. Returns (chi2_statistic, pvalue).

    A p-value below ~0.001 is the conventional alarm threshold for SRM:
    the assignment split deviates from intended ratios more than chance allows.
    """
    counts = np.asarray(observed_counts, dtype=float)
    total = counts.sum()
    if expected_ratio is None:
        expected = np.full_like(counts, total / len(counts))
    else:
        ratios = np.asarray(expected_ratio, dtype=float)
        expected = ratios / ratios.sum() * total
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    pvalue = float(1 - stats.chi2.cdf(chi2, df=len(counts) - 1))
    return chi2, pvalue

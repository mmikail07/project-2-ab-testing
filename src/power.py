"""Sample-size and power analysis. Analytical formulas via statsmodels plus a
Monte Carlo cross-check so the notebook can show both methods agreeing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerResult:
    method: str
    n_per_arm: int
    baseline: float
    treatment: float
    mde_pp: float
    alpha: float
    power: float


def analytical_sample_size(
    baseline: float,
    mde_pp: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> PowerResult:
    treatment = baseline + mde_pp / 100.0
    effect = proportion_effectsize(treatment, baseline)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=alternative,
    )
    return PowerResult(
        method="analytical (statsmodels.NormalIndPower)",
        n_per_arm=int(np.ceil(n)),
        baseline=float(baseline),
        treatment=float(treatment),
        mde_pp=float(mde_pp),
        alpha=float(alpha),
        power=float(power),
    )


def simulate_power(
    baseline: float,
    mde_pp: float,
    n_per_arm: int,
    n_sims: int = 5_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PowerResult:
    """Empirical power via Monte Carlo. Runs n_sims experiments at the given
    sample size with the true treatment effect baked in, counts how many reject H0.
    """
    rng = np.random.default_rng(seed)
    treatment = baseline + mde_pp / 100.0
    if not 0 <= treatment <= 1:
        raise ValueError(f"treatment rate {treatment} outside [0,1]")

    successes_c = rng.binomial(n_per_arm, baseline, size=n_sims)
    successes_t = rng.binomial(n_per_arm, treatment, size=n_sims)
    p_c = successes_c / n_per_arm
    p_t = successes_t / n_per_arm
    p_pool = (successes_c + successes_t) / (2 * n_per_arm)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per_arm)
    se = np.where(se == 0, 1e-12, se)
    z = (p_t - p_c) / se
    from scipy.stats import norm
    pvalues = 2 * (1 - norm.cdf(np.abs(z)))
    empirical_power = float((pvalues < alpha).mean())

    return PowerResult(
        method=f"simulation (n_sims={n_sims})",
        n_per_arm=int(n_per_arm),
        baseline=float(baseline),
        treatment=float(treatment),
        mde_pp=float(mde_pp),
        alpha=float(alpha),
        power=empirical_power,
    )


def power_curve(
    baseline: float,
    mde_pp: float,
    n_grid: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Analytical power as a function of sample size per arm. Vectorized."""
    treatment = baseline + mde_pp / 100.0
    effect = proportion_effectsize(treatment, baseline)
    analysis = NormalIndPower()
    return np.array([
        analysis.power(effect_size=effect, nobs1=int(n), alpha=alpha, ratio=1.0, alternative="two-sided")
        for n in n_grid
    ])


def mde_curve(
    baseline: float,
    n_grid: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> np.ndarray:
    """Detectable lift in pp as a function of sample size per arm. Solves for
    effect size at each N and converts back to a pp delta."""
    analysis = NormalIndPower()
    mdes = []
    for n in n_grid:
        eff = analysis.solve_power(
            effect_size=None,
            nobs1=int(n),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
        treatment = _proportion_from_effectsize(baseline, eff)
        mdes.append((treatment - baseline) * 100.0)
    return np.array(mdes)


def _proportion_from_effectsize(p1: float, effect: float) -> float:
    """Inverse of proportion_effectsize: given p1 and Cohen's h, recover p2."""
    return float(np.sin(np.arcsin(np.sqrt(p1)) + effect / 2.0) ** 2)

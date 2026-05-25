"""Bayesian companion analysis: Beta-Binomial conjugate posterior on conversion.

Frequentist analysis answers 'is the lift significant at alpha=0.05?'. Bayesian
analysis answers 'what is P(treatment > control | data)?' and 'what is the
expected loss of choosing the wrong arm?'. Stakeholders often find the second
framing easier to act on.

The model is conjugate: Beta(alpha_0, beta_0) prior on each arm's conversion
rate, updated to Beta(alpha_0 + successes, beta_0 + failures) posterior. P(theta_t > theta_c)
is computed by direct Monte Carlo on the posteriors (closed-form solutions exist
but the MC version is more transparent for the notebook).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class PosteriorComparison:
    successes_c: int
    n_c: int
    successes_t: int
    n_t: int
    prior_alpha: float
    prior_beta: float
    mean_c: float
    mean_t: float
    prob_t_greater_than_c: float
    ci_low_lift: float
    ci_high_lift: float
    expected_loss_choose_c: float
    expected_loss_choose_t: float
    samples_c: np.ndarray
    samples_t: np.ndarray


def beta_posterior_params(successes: int, n: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> tuple[float, float]:
    """Beta-Binomial conjugate update. Beta(1, 1) is the uniform prior on [0, 1]."""
    return (prior_alpha + successes, prior_beta + (n - successes))


def sample_posterior(successes: int, n: int, n_samples: int = 100_000, prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int = 42) -> np.ndarray:
    a, b = beta_posterior_params(successes, n, prior_alpha, prior_beta)
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, size=n_samples)


def compare_arms(
    successes_c: int,
    n_c: int,
    successes_t: int,
    n_t: int,
    n_samples: int = 100_000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
    seed: int = 42,
) -> PosteriorComparison:
    samples_c = sample_posterior(successes_c, n_c, n_samples, prior_alpha, prior_beta, seed=seed)
    samples_t = sample_posterior(successes_t, n_t, n_samples, prior_alpha, prior_beta, seed=seed + 1)
    lift_samples = samples_t - samples_c
    a_low = (1 - credible_level) / 2
    a_high = 1 - a_low
    ci_low, ci_high = np.quantile(lift_samples, [a_low, a_high])

    return PosteriorComparison(
        successes_c=int(successes_c),
        n_c=int(n_c),
        successes_t=int(successes_t),
        n_t=int(n_t),
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        mean_c=float(samples_c.mean()),
        mean_t=float(samples_t.mean()),
        prob_t_greater_than_c=float((samples_t > samples_c).mean()),
        ci_low_lift=float(ci_low),
        ci_high_lift=float(ci_high),
        expected_loss_choose_c=float(np.maximum(samples_t - samples_c, 0).mean()),
        expected_loss_choose_t=float(np.maximum(samples_c - samples_t, 0).mean()),
        samples_c=samples_c,
        samples_t=samples_t,
    )


def credible_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a_low = (1 - level) / 2
    a_high = 1 - a_low
    return tuple(np.quantile(samples, [a_low, a_high]))


def posterior_density(samples: np.ndarray, grid: np.ndarray | None = None, bw: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Kernel density estimate of a posterior sample, returned as (grid, density)."""
    if grid is None:
        lo = np.quantile(samples, 0.001)
        hi = np.quantile(samples, 0.999)
        grid = np.linspace(lo, hi, 400)
    kde = scipy_stats.gaussian_kde(samples, bw_method=bw if bw is None else None)
    return grid, kde(grid)

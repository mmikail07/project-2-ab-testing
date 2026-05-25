"""Tests for src.bayes."""
from __future__ import annotations

import numpy as np

from src.bayes import (
    beta_posterior_params,
    compare_arms,
    credible_interval,
    sample_posterior,
)


def test_beta_posterior_params_uniform_prior():
    a, b = beta_posterior_params(successes=10, n=100, prior_alpha=1, prior_beta=1)
    assert a == 11
    assert b == 91


def test_sample_posterior_mean_matches_observed_rate():
    samples = sample_posterior(successes=500, n=10_000, n_samples=20_000, seed=1)
    assert abs(samples.mean() - 0.05) < 0.005


def test_compare_arms_recovers_positive_lift_with_high_probability():
    """When treatment really beats control, P(T > C) should be near 1."""
    result = compare_arms(
        successes_c=500, n_c=10_000,
        successes_t=700, n_t=10_000,
        n_samples=20_000, seed=5,
    )
    assert result.prob_t_greater_than_c > 0.99
    assert result.ci_low_lift > 0


def test_compare_arms_uncertain_when_arms_are_equal():
    """Identical successes should give P(T > C) ~ 0.5."""
    result = compare_arms(
        successes_c=500, n_c=10_000,
        successes_t=500, n_t=10_000,
        n_samples=20_000, seed=6,
    )
    assert 0.45 <= result.prob_t_greater_than_c <= 0.55


def test_compare_arms_credible_interval_contains_true_lift():
    """With clear data, the 95% credible interval on lift should contain
    the observed difference."""
    result = compare_arms(
        successes_c=500, n_c=10_000,
        successes_t=700, n_t=10_000,
        seed=7,
    )
    obs_lift = 700 / 10_000 - 500 / 10_000
    assert result.ci_low_lift < obs_lift < result.ci_high_lift


def test_expected_loss_is_zero_when_choice_is_clearly_correct():
    """If treatment is overwhelmingly better, expected loss of choosing
    treatment should be near zero."""
    result = compare_arms(
        successes_c=300, n_c=10_000,
        successes_t=900, n_t=10_000,
        seed=8,
    )
    assert result.expected_loss_choose_t < 0.001
    assert result.expected_loss_choose_c > 0.005


def test_credible_interval_basic():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.05, 0.01, size=10_000)
    low, high = credible_interval(samples, level=0.95)
    assert low < 0.05 < high
    assert high - low < 0.05

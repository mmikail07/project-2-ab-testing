"""Tests for src.power. Analytical and simulation-based methods must agree."""
from __future__ import annotations

from src.power import analytical_sample_size, mde_curve, power_curve, simulate_power
import numpy as np


def test_analytical_sample_size_for_known_case():
    """Baseline 12%, MDE 2pp, alpha 0.05, power 0.80 via Cohen's h (statsmodels'
    NormalIndPower) gives ~4400 per arm. Sources vary by ±10% depending on whether
    pooled or unpooled SE is used in the analytical derivation."""
    result = analytical_sample_size(baseline=0.12, mde_pp=2.0)
    assert 3_900 <= result.n_per_arm <= 4_900


def test_higher_mde_needs_smaller_sample():
    small = analytical_sample_size(baseline=0.12, mde_pp=2.0)
    large = analytical_sample_size(baseline=0.12, mde_pp=4.0)
    assert large.n_per_arm < small.n_per_arm


def test_higher_power_needs_larger_sample():
    eighty = analytical_sample_size(baseline=0.12, mde_pp=2.0, power=0.80)
    ninety = analytical_sample_size(baseline=0.12, mde_pp=2.0, power=0.90)
    assert ninety.n_per_arm > eighty.n_per_arm


def test_simulation_power_matches_analytical_within_2pp():
    """The headline cross-check: simulate at the analytical sample size and
    expect empirical power within 2pp of the target."""
    analytical = analytical_sample_size(baseline=0.12, mde_pp=2.0, power=0.80)
    sim = simulate_power(
        baseline=0.12,
        mde_pp=2.0,
        n_per_arm=analytical.n_per_arm,
        n_sims=3_000,
        seed=1,
    )
    assert abs(sim.power - 0.80) < 0.02, f"simulated power {sim.power:.3f} drifted from 0.80"


def test_power_curve_monotonic_in_n():
    n_grid = np.array([500, 1_000, 2_000, 4_000, 8_000])
    powers = power_curve(baseline=0.12, mde_pp=2.0, n_grid=n_grid)
    assert np.all(np.diff(powers) > 0)


def test_mde_curve_monotonic_decreasing():
    n_grid = np.array([500, 1_000, 2_000, 4_000, 8_000])
    mdes = mde_curve(baseline=0.12, n_grid=n_grid)
    assert np.all(np.diff(mdes) < 0)

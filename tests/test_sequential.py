"""Tests for src.sequential. The peeking story has to be empirically demonstrable:
naive interim looks inflate FPR; OBF and mSPRT control it."""
from __future__ import annotations

import numpy as np
import pytest

from src.sequential import obrien_fleming_bounds, simulate_peeking


def test_naive_peeking_inflates_fpr():
    """4 peeks at alpha=0.05 should push FPR well above 0.05."""
    result = simulate_peeking(
        n_per_arm=5_000, baseline=0.05,
        method="naive", n_sims=5_000, seed=1,
    )
    assert result.empirical_fpr > 0.10, (
        f"naive peeking should inflate FPR substantially; got {result.empirical_fpr:.4f}"
    )


def test_obf_controls_fpr_near_alpha():
    """OBF bounds should bring FPR back near alpha."""
    result = simulate_peeking(
        n_per_arm=5_000, baseline=0.05,
        method="obf", n_sims=5_000, seed=2,
    )
    assert 0.03 <= result.empirical_fpr <= 0.07, (
        f"OBF should control FPR near 0.05; got {result.empirical_fpr:.4f}"
    )


def test_mspr_controls_fpr_at_or_below_alpha():
    """mSPRT is always-valid; FPR <= alpha across any peeking pattern."""
    result = simulate_peeking(
        n_per_arm=5_000, baseline=0.05,
        method="mspr", n_sims=5_000, seed=3,
    )
    assert result.empirical_fpr <= 0.07, (
        f"mSPRT should hold FPR at or below alpha; got {result.empirical_fpr:.4f}"
    )


def test_obrien_fleming_critical_z_at_final_look_close_to_constant():
    """At the final look (k=K), z_K = c. Earlier looks should have larger thresholds."""
    bounds = obrien_fleming_bounds(n_peeks=4, alpha=0.05)
    assert bounds[-1] < bounds[0]
    assert abs(bounds[-1] - 2.024) < 1e-6
    assert (np.diff(bounds) < 0).all()


def test_obrien_fleming_raises_for_unsupported_settings():
    with pytest.raises(ValueError):
        obrien_fleming_bounds(n_peeks=7, alpha=0.05)


def test_rejection_share_at_peek_is_monotone():
    """Cumulative rejection share can only go up across peeks."""
    result = simulate_peeking(n_per_arm=3_000, baseline=0.05, method="naive", n_sims=2_000, seed=4)
    assert (np.diff(result.rejection_share_at_peek) >= 0).all()


def test_simulate_peeking_rejects_unknown_method():
    with pytest.raises(ValueError):
        simulate_peeking(n_per_arm=1_000, baseline=0.05, method="bogus", n_sims=100, seed=0)

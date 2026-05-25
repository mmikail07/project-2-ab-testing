"""Tests for src.simulate. Reproducibility, schema, and ground-truth recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.simulate import generate_experiment, summarize


def test_reproducibility_same_seed_same_data():
    df1, t1 = generate_experiment(n_users=2_000, lift_pp=2.5, seed=7)
    df2, t2 = generate_experiment(n_users=2_000, lift_pp=2.5, seed=7)
    pd.testing.assert_frame_equal(df1, df2)
    assert t1 == t2


def test_different_seeds_diverge():
    df1, _ = generate_experiment(n_users=2_000, lift_pp=2.5, seed=7)
    df2, _ = generate_experiment(n_users=2_000, lift_pp=2.5, seed=8)
    assert not df1.equals(df2)


def test_schema_columns_present():
    df, _ = generate_experiment(n_users=500, seed=1)
    required = {
        "user_id", "arm", "is_returning", "is_mobile", "high_aov",
        "impression", "product_view", "add_to_cart", "checkout_start",
        "payment", "completion", "aov_aed", "items_per_order",
        "page_load_ms", "refunded", "cart_abandoned",
    }
    assert required.issubset(df.columns)
    assert df["arm"].isin(["control", "treatment"]).all()


def test_funnel_monotonic_no_completion_without_payment():
    df, _ = generate_experiment(n_users=5_000, seed=3)
    assert (df.loc[df["completion"], "payment"]).all()
    assert (df.loc[df["payment"], "checkout_start"]).all()
    assert (df.loc[df["checkout_start"], "add_to_cart"]).all()
    assert (df.loc[df["add_to_cart"], "product_view"]).all()


def test_secondary_metrics_only_for_completers():
    df, _ = generate_experiment(n_users=5_000, seed=3)
    assert df.loc[~df["completion"], "aov_aed"].isna().all()
    assert df.loc[df["completion"], "aov_aed"].notna().all()


def test_lift_recovered_within_tolerance():
    """At n=200k, the observed lift should be within 0.5pp of the analytical 2.5pp."""
    df, truth = generate_experiment(n_users=200_000, lift_pp=2.5, seed=42)
    rate_c = df.loc[df["arm"] == "control", "completion"].mean()
    rate_t = df.loc[df["arm"] == "treatment", "completion"].mean()
    observed_lift_pp = (rate_t - rate_c) * 100
    assert abs(observed_lift_pp - truth.true_lift_pp) < 0.5, (
        f"observed lift {observed_lift_pp:.3f}pp deviated from true {truth.true_lift_pp}pp"
    )


def test_arm_split_close_to_target():
    df, _ = generate_experiment(n_users=10_000, seed=42, arm_split=0.5)
    share_control = (df["arm"] == "control").mean()
    assert 0.48 <= share_control <= 0.52


def test_summarize_columns():
    df, _ = generate_experiment(n_users=2_000, seed=2)
    s = summarize(df)
    assert {"arm", "n", "completions", "completion_rate"}.issubset(s.columns)
    assert len(s) == 2


def test_baseline_completion_rate_matches_chain():
    """The analytical helper should match the product of conditional rates."""
    baseline = config.load_funnel_baseline()
    expected = np.prod([s["conditional_rate"] for s in baseline["stages"]])
    assert abs(config.baseline_completion_rate() - expected) < 1e-12

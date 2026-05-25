"""Tests for src.guardrails."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.guardrails import all_guardrails, check_cart_abandonment, check_page_load, check_refund_rate


def _events_with_guardrail_breach(n=20_000, refund_rate_inflation: float = 0.0, seed=1):
    rng = np.random.default_rng(seed)
    arm = np.where(rng.random(n) < 0.5, "control", "treatment")
    completion = rng.random(n) < 0.05
    refund_rate = np.where(arm == "treatment", 0.04 + refund_rate_inflation, 0.04)
    refunded = completion & (rng.random(n) < refund_rate)
    cart_abandoned = (rng.random(n) < 0.4) & ~completion
    page_load = rng.normal(1800, 100, size=n)
    return pd.DataFrame({
        "arm": arm,
        "completion": completion,
        "refunded": refunded,
        "cart_abandoned": cart_abandoned,
        "page_load_ms": page_load,
    })


def test_refund_guardrail_does_not_fire_when_no_inflation():
    df = _events_with_guardrail_breach(refund_rate_inflation=0.0, n=30_000)
    result = check_refund_rate(df)
    assert not result.flagged
    assert result.pvalue > 0.05


def test_refund_guardrail_fires_on_severe_inflation():
    """If treatment's per-order refund rate jumps by 10pp, the guardrail must trigger."""
    df = _events_with_guardrail_breach(refund_rate_inflation=0.10, n=30_000)
    result = check_refund_rate(df)
    assert result.flagged
    assert result.pvalue < 0.05
    assert result.treatment_value > result.control_value


def test_refund_guardrail_uses_per_order_denominator():
    """Sanity: with the same per-order refund rate but more orders in treatment,
    the per-order metric should not fire even though the raw rate does."""
    rng = np.random.default_rng(7)
    n = 40_000
    arm = np.where(rng.random(n) < 0.5, "control", "treatment")
    completion = np.where(arm == "treatment", rng.random(n) < 0.10, rng.random(n) < 0.05)
    refunded = completion & (rng.random(n) < 0.04)
    df = pd.DataFrame({
        "arm": arm, "completion": completion, "refunded": refunded,
        "cart_abandoned": np.zeros(n, dtype=bool),
        "page_load_ms": rng.normal(1800, 100, size=n),
    })
    result = check_refund_rate(df)
    assert not result.flagged


def test_page_load_guardrail_one_sided():
    """Treatment page_load same as control → no alarm. Treatment slower → alarm."""
    rng = np.random.default_rng(11)
    n = 20_000
    arm = np.where(rng.random(n) < 0.5, "control", "treatment")
    # treatment 300ms slower
    page_load = np.where(arm == "treatment", rng.normal(2100, 100, size=n), rng.normal(1800, 100, size=n))
    df = pd.DataFrame({
        "arm": arm,
        "completion": np.zeros(n, dtype=bool),
        "refunded": np.zeros(n, dtype=bool),
        "cart_abandoned": np.zeros(n, dtype=bool),
        "page_load_ms": page_load,
    })
    result = check_page_load(df)
    assert result.flagged
    assert result.delta > 0


def test_all_guardrails_returns_frame_with_expected_metrics():
    df = _events_with_guardrail_breach(n=10_000)
    table = all_guardrails(df)
    assert {"metric", "control", "treatment", "delta", "pvalue_one_sided", "flagged"}.issubset(table.columns)
    metrics = set(table["metric"])
    assert "refund_rate_per_order" in metrics
    assert "cart_abandonment_rate" in metrics
    assert "page_load_ms_mean" in metrics

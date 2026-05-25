"""Guardrail metrics: one-sided directional checks for adverse treatment moves.

A primary lift means nothing if a guardrail moves the wrong way. Refund rate
up by 8% on a 3% conversion lift is an obvious example: ship the treatment
and you lose money even though the test 'won'. Guardrails are the safety net.

Each check is one-sided in the *adverse* direction. We do not care if page
loads got faster or refunds dropped; we only flag when they got worse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import two_proportion_ztest, welch_ttest


@dataclass
class GuardrailResult:
    name: str
    direction_adverse: str
    control_value: float
    treatment_value: float
    delta: float
    pvalue: float
    flagged: bool


def check_refund_rate(events: pd.DataFrame, alpha: float = 0.05) -> GuardrailResult:
    """Per-order refund rate going up is bad. Conditioned on completion so the
    metric measures order quality, not order volume. Without this conditioning
    a primary-metric win would mechanically inflate the raw refund rate."""
    completed = events[events["completion"]]
    control = completed[completed["arm"] == "control"]
    treatment = completed[completed["arm"] == "treatment"]
    result = two_proportion_ztest(
        successes_c=int(control["refunded"].sum()),
        n_c=int(len(control)),
        successes_t=int(treatment["refunded"].sum()),
        n_t=int(len(treatment)),
        alpha=alpha,
        alternative="larger",
    )
    return GuardrailResult(
        name="refund_rate_per_order",
        direction_adverse="increase",
        control_value=float(control["refunded"].mean()),
        treatment_value=float(treatment["refunded"].mean()),
        delta=float(result.point_estimate),
        pvalue=float(result.pvalue),
        flagged=bool(result.pvalue < alpha),
    )


def check_cart_abandonment(events: pd.DataFrame, alpha: float = 0.05) -> GuardrailResult:
    """Cart abandonment up is bad."""
    control = events[events["arm"] == "control"]
    treatment = events[events["arm"] == "treatment"]
    result = two_proportion_ztest(
        successes_c=int(control["cart_abandoned"].sum()),
        n_c=int(len(control)),
        successes_t=int(treatment["cart_abandoned"].sum()),
        n_t=int(len(treatment)),
        alpha=alpha,
        alternative="larger",
    )
    return GuardrailResult(
        name="cart_abandonment_rate",
        direction_adverse="increase",
        control_value=float(control["cart_abandoned"].mean()),
        treatment_value=float(treatment["cart_abandoned"].mean()),
        delta=float(result.point_estimate),
        pvalue=float(result.pvalue),
        flagged=bool(result.pvalue < alpha),
    )


def check_page_load(events: pd.DataFrame, alpha: float = 0.05) -> GuardrailResult:
    """Page load going up (slower) is bad. Welch's t on the continuous metric."""
    control = events.loc[events["arm"] == "control", "page_load_ms"].to_numpy()
    treatment = events.loc[events["arm"] == "treatment", "page_load_ms"].to_numpy()
    result = welch_ttest(control, treatment, alpha=alpha)
    one_sided_p = result.pvalue / 2 if result.point_estimate > 0 else 1 - result.pvalue / 2
    return GuardrailResult(
        name="page_load_ms_mean",
        direction_adverse="increase",
        control_value=float(np.nanmean(control)),
        treatment_value=float(np.nanmean(treatment)),
        delta=float(result.point_estimate),
        pvalue=float(one_sided_p),
        flagged=bool(one_sided_p < alpha and result.point_estimate > 0),
    )


def all_guardrails(events: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    results = [
        check_refund_rate(events, alpha),
        check_cart_abandonment(events, alpha),
        check_page_load(events, alpha),
    ]
    return pd.DataFrame([{
        "metric": r.name,
        "adverse_direction": r.direction_adverse,
        "control": r.control_value,
        "treatment": r.treatment_value,
        "delta": r.delta,
        "pvalue_one_sided": r.pvalue,
        "flagged": r.flagged,
    } for r in results])

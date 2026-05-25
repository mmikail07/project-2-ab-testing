"""Synthetic A/B experiment generator with recorded ground truth.

Produces a user-level event log: one row per user with arm assignment,
segment flags, six funnel-stage booleans, and secondary / guardrail metrics.
The control arm uses funnel conditional rates from data/external/funnel_baseline.json
unchanged. Treatment users receive an additional Bernoulli "rescue" draw so the
marginal completion rate increases by exactly lift_pp percentage points in
expectation. Rescued users have all upstream stages flipped to True so the
event log stays internally consistent (no completion without a payment).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config
from .io_utils import save_events, save_manifest


@dataclass
class GroundTruth:
    n_users: int
    seed: int
    arm_split: float
    true_lift_pp: float
    analytical_baseline_rate: float
    analytical_target_rate: float
    additional_completion_prob: float
    treatment_stage: str
    funnel_baseline_path: str


def generate_experiment(
    n_users: int = config.DEFAULT_N_USERS,
    lift_pp: float = config.DEFAULT_TRUE_LIFT_PP,
    seed: int = config.SEED,
    arm_split: float = 0.5,
) -> tuple[pd.DataFrame, GroundTruth]:
    rng = np.random.default_rng(seed)
    baseline = config.load_funnel_baseline()
    stages = baseline["stages"]
    stage_names = [s["name"] for s in stages]

    user_id = np.arange(1, n_users + 1, dtype=np.int64)
    arm = np.where(rng.random(n_users) < arm_split, "control", "treatment")
    treatment_mask = arm == "treatment"

    seg_cfg = baseline["segments"]
    is_returning = rng.random(n_users) < seg_cfg["is_returning"]["share"]
    is_mobile = rng.random(n_users) < seg_cfg["is_mobile"]["share"]
    high_aov = rng.random(n_users) < seg_cfg["high_aov"]["share"]

    stage_flags: dict[str, np.ndarray] = {}
    survived = np.ones(n_users, dtype=bool)
    for s in stages:
        name = s["name"]
        if name == "impression":
            flag = np.ones(n_users, dtype=bool)
        else:
            flag = survived & (rng.random(n_users) < s["conditional_rate"])
        stage_flags[name] = flag
        survived = flag

    analytical_baseline = config.baseline_completion_rate()
    analytical_target = analytical_baseline + lift_pp / 100.0
    p_didnt = 1.0 - analytical_baseline
    if p_didnt <= 0:
        raise ValueError("baseline completion rate >= 1, cannot add positive lift")
    additional = (lift_pp / 100.0) / p_didnt
    if not 0.0 <= additional <= 1.0:
        raise ValueError(
            f"lift_pp={lift_pp} produces additional_prob={additional:.4f} outside [0,1]; "
            f"reduce lift_pp or recalibrate funnel_baseline.json"
        )

    completion_control = stage_flags["completion"]
    rescue_draw = rng.random(n_users) < additional
    rescue_mask = treatment_mask & (~completion_control) & rescue_draw

    for name in stage_names:
        stage_flags[name] = stage_flags[name] | rescue_mask

    sec = baseline["secondary_metrics"]
    completed = stage_flags["completion"]
    aov_raw = rng.normal(sec["aov_aed_mean_control"], sec["aov_aed_sd_control"], size=n_users)
    aov = np.where(completed, np.clip(aov_raw, 0.0, None), np.nan)
    items_raw = rng.normal(sec["items_per_order_mean_control"], sec["items_per_order_sd_control"], size=n_users)
    items = np.where(completed, np.clip(items_raw, 1.0, None).round(), np.nan)

    g = baseline["guardrail_metrics"]
    page_load_ms = np.clip(
        rng.normal(g["page_load_p95_ms_control"], g["page_load_p95_ms_sd"], size=n_users),
        200.0,
        None,
    )
    refunded = completed & (rng.random(n_users) < g["refund_rate_control"])
    cart_abandoned = stage_flags["add_to_cart"] & (~completed)

    df = pd.DataFrame({
        "user_id": user_id,
        "arm": arm,
        "is_returning": is_returning,
        "is_mobile": is_mobile,
        "high_aov": high_aov,
        **{name: stage_flags[name] for name in stage_names},
        "aov_aed": aov,
        "items_per_order": items,
        "page_load_ms": page_load_ms,
        "refunded": refunded,
        "cart_abandoned": cart_abandoned,
    })

    truth = GroundTruth(
        n_users=int(n_users),
        seed=int(seed),
        arm_split=float(arm_split),
        true_lift_pp=float(lift_pp),
        analytical_baseline_rate=float(analytical_baseline),
        analytical_target_rate=float(analytical_target),
        additional_completion_prob=float(additional),
        treatment_stage="completion",
        funnel_baseline_path=str(config.FUNNEL_BASELINE_JSON.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
    )
    return df, truth


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby("arm").agg(
        n=("user_id", "count"),
        product_views=("product_view", "sum"),
        add_to_carts=("add_to_cart", "sum"),
        checkout_starts=("checkout_start", "sum"),
        payments=("payment", "sum"),
        completions=("completion", "sum"),
        mean_aov=("aov_aed", "mean"),
        mean_items=("items_per_order", "mean"),
        mean_page_load_ms=("page_load_ms", "mean"),
        refund_rate=("refunded", "mean"),
    )
    g["completion_rate"] = g["completions"] / g["n"]
    return g.reset_index()


def main() -> None:
    events, truth = generate_experiment()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_events(events, config.EXPERIMENT_EVENTS_PARQUET)
    summarize(events).to_csv(config.EXPERIMENT_SUMMARY_CSV, index=False)
    save_manifest(asdict(truth), config.GROUND_TRUTH_JSON)
    print(f"wrote {len(events)} rows to {config.EXPERIMENT_EVENTS_PARQUET.name}")
    print(f"analytical baseline rate: {truth.analytical_baseline_rate:.4f}")
    print(f"analytical target rate:   {truth.analytical_target_rate:.4f}")
    print(f"true lift (pp):           {truth.true_lift_pp:.2f}")


if __name__ == "__main__":
    main()

"""Sidebar inputs and result-card components for the calculator."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class CalculatorInputs:
    baseline: float
    mde_pp: float
    alpha: float
    power: float
    daily_traffic_per_arm: int
    mde_mode: str
    relative_mde_pct: float


def sidebar_inputs() -> CalculatorInputs:
    st.sidebar.header("Inputs")

    baseline_pct = st.sidebar.slider(
        "Baseline conversion rate (%)",
        min_value=0.5, max_value=30.0, value=5.0, step=0.5,
        help="Current conversion rate of the metric you are testing.",
    )
    baseline = baseline_pct / 100.0

    mde_mode = st.sidebar.radio(
        "MDE expressed as",
        options=["absolute (pp)", "relative (%)"],
        horizontal=True,
    )

    if mde_mode == "absolute (pp)":
        mde_pp = st.sidebar.slider(
            "Minimum detectable effect (pp)",
            min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            help="Smallest absolute lift that would justify shipping.",
        )
        relative_mde_pct = mde_pp / baseline_pct * 100.0
    else:
        relative_mde_pct = st.sidebar.slider(
            "Minimum detectable effect (% relative)",
            min_value=1.0, max_value=100.0, value=40.0, step=1.0,
            help="Smallest relative lift that would justify shipping.",
        )
        mde_pp = baseline_pct * relative_mde_pct / 100.0

    alpha = st.sidebar.select_slider(
        "Alpha",
        options=[0.01, 0.025, 0.05, 0.10],
        value=0.05,
        help="Type I error rate (false positive rate you accept).",
    )

    power = st.sidebar.select_slider(
        "Power",
        options=[0.70, 0.80, 0.90, 0.95],
        value=0.80,
        help="Probability of detecting the lift if it is real.",
    )

    daily_traffic = st.sidebar.number_input(
        "Daily traffic per arm",
        min_value=100, max_value=10_000_000, value=25_000, step=1_000,
        help="Unique users per day per arm hitting the experimental surface.",
    )

    return CalculatorInputs(
        baseline=baseline,
        mde_pp=mde_pp,
        alpha=alpha,
        power=power,
        daily_traffic_per_arm=int(daily_traffic),
        mde_mode=mde_mode,
        relative_mde_pct=relative_mde_pct,
    )


def result_cards(n_per_arm: int, days: float, baseline: float, mde_pp: float, relative_mde_pct: float) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="Required sample size per arm",
            value=f"{n_per_arm:,}",
            help="Round up. Total experiment size is 2x this.",
        )
        st.metric(
            label="Total users (both arms)",
            value=f"{n_per_arm * 2:,}",
        )
    with c2:
        st.metric(
            label="Estimated duration",
            value=f"{days:.1f} days",
            help="At your specified daily traffic per arm.",
        )
        st.metric(
            label="Baseline → required treatment rate",
            value=f"{baseline*100:.2f}% → {(baseline + mde_pp/100)*100:.2f}%",
            help=f"Relative lift: {relative_mde_pct:.1f}%",
        )

"""A/B Test Sample-Size Calculator — Streamlit entry point.

Deployed on Streamlit Community Cloud. Same statistical machinery as the
notebooks (src.power, src.preflight) so calculator and analysis agree.

Run locally:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.power import analytical_sample_size, mde_curve, power_curve  # noqa: E402
from src.preflight import check_srm  # noqa: E402

from streamlit_app.components import result_cards, sidebar_inputs  # noqa: E402
from streamlit_app.plots import mde_curve_chart, power_curve_chart  # noqa: E402


st.set_page_config(
    page_title="A/B Test Sample-Size Calculator",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_power_curve(baseline: float, mde_pp: float, alpha: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    n_grid = np.linspace(500, n_max, 80, dtype=int)
    powers = power_curve(baseline=baseline, mde_pp=mde_pp, n_grid=n_grid, alpha=alpha)
    return n_grid, powers


@st.cache_data(show_spinner=False)
def cached_mde_curve(baseline: float, alpha: float, power: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    n_grid = np.linspace(500, n_max, 60, dtype=int)
    mdes = mde_curve(baseline=baseline, n_grid=n_grid, alpha=alpha, power=power)
    return n_grid, mdes


st.title("A/B Test Sample-Size Calculator")
st.caption(
    "Analytical sample size and duration estimate for two-proportion experiments. "
    "Powered by statsmodels' NormalIndPower under the hood. Same code path used in this repo's notebooks."
)

inputs = sidebar_inputs()

result = analytical_sample_size(
    baseline=inputs.baseline,
    mde_pp=inputs.mde_pp,
    alpha=inputs.alpha,
    power=inputs.power,
)
days = result.n_per_arm / inputs.daily_traffic_per_arm

st.subheader("Headline")
result_cards(
    n_per_arm=result.n_per_arm,
    days=days,
    baseline=inputs.baseline,
    mde_pp=inputs.mde_pp,
    relative_mde_pct=inputs.relative_mde_pct,
)

st.divider()

st.subheader("Power vs sample size")
st.caption(f"At MDE = {inputs.mde_pp:.2f}pp, alpha = {inputs.alpha}.")
n_max = max(result.n_per_arm * 3, 5_000)
n_grid, powers = cached_power_curve(inputs.baseline, inputs.mde_pp, inputs.alpha, n_max)
st.plotly_chart(
    power_curve_chart(n_grid, powers, target_n=result.n_per_arm, target_power=inputs.power),
    width="stretch",
)

st.subheader("Detectable lift vs sample size")
st.caption(f"At power = {inputs.power}, alpha = {inputs.alpha}.")
n_grid_mde, mdes = cached_mde_curve(inputs.baseline, inputs.alpha, inputs.power, n_max)
st.plotly_chart(
    mde_curve_chart(n_grid_mde, mdes, target_n=result.n_per_arm, target_mde_pp=inputs.mde_pp),
    width="stretch",
)

st.divider()

with st.expander("SRM mini-checker", expanded=False):
    st.markdown(
        "Paste your final arm counts to chi-square test against the intended split. "
        "A p-value below 0.001 is the conventional alarm threshold."
    )
    c1, c2 = st.columns(2)
    with c1:
        control_count = st.number_input("Control arm count", min_value=1, value=25_000, step=100)
    with c2:
        treatment_count = st.number_input("Treatment arm count", min_value=1, value=25_000, step=100)
    if st.button("Run SRM check"):
        srm = check_srm({"control": int(control_count), "treatment": int(treatment_count)})
        col1, col2, col3 = st.columns(3)
        col1.metric("chi-square", f"{srm['chi2']:.2f}")
        col2.metric("p-value", f"{srm['pvalue']:.4g}")
        col3.metric("verdict", srm["verdict"].upper())
        if srm["verdict"] == "alarm":
            st.error("SRM ALARM. Assignment ratio deviates from 50/50 more than chance allows. Do not analyze.")
        else:
            st.success("SRM check passes. Assignment ratio is within chance variation.")

with st.expander("How this calculator works", expanded=False):
    copy_path = Path(__file__).parent / "copy.md"
    if copy_path.exists():
        st.markdown(copy_path.read_text(encoding="utf-8"))

st.divider()
st.caption(
    f"Project 2 of Mohammad Mikail's UAE data-science portfolio. "
    f"Source: [github.com/mmikail07/project-2-ab-testing](https://github.com/mmikail07/project-2-ab-testing). "
    f"Built with statsmodels, scipy, plotly, and Streamlit."
)

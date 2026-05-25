"""Plotly chart builders for the Streamlit sample-size calculator."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def power_curve_chart(n_grid: np.ndarray, power_values: np.ndarray, target_n: int, target_power: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_grid, y=power_values,
        mode="lines",
        line=dict(width=3, color="#4c72b0"),
        name="power",
        hovertemplate="n=%{x:,}<br>power=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=target_power, line=dict(dash="dash", color="#999"),
        annotation_text=f"target power = {target_power:.2f}", annotation_position="top left",
    )
    fig.add_vline(
        x=target_n, line=dict(dash="dash", color="#999"),
        annotation_text=f"required n = {target_n:,}", annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="sample size per arm",
        yaxis_title="statistical power",
        yaxis_range=[0, 1.02],
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=20, b=40),
    )
    return fig


def mde_curve_chart(n_grid: np.ndarray, mde_values_pp: np.ndarray, target_n: int, target_mde_pp: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_grid, y=mde_values_pp,
        mode="lines",
        line=dict(width=3, color="#55a868"),
        name="detectable lift",
        hovertemplate="n=%{x:,}<br>MDE=%{y:.3f}pp<extra></extra>",
    ))
    fig.add_hline(
        y=target_mde_pp, line=dict(dash="dash", color="#999"),
        annotation_text=f"target MDE = {target_mde_pp:.2f}pp", annotation_position="top right",
    )
    fig.add_vline(
        x=target_n, line=dict(dash="dash", color="#999"),
        annotation_text=f"n = {target_n:,}", annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="sample size per arm",
        yaxis_title="detectable lift (percentage points)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=20, b=40),
    )
    return fig

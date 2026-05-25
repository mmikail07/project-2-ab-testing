"""Reusable matplotlib figure builders. Notebooks import these so chart style
stays consistent across the repo and the README screenshots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def funnel_plot(summary: pd.DataFrame, save_to: Path | None = None) -> plt.Figure:
    """Horizontal funnel bar by arm. Expects the output of src.simulate.summarize."""
    _style()
    stages = ["product_views", "add_to_carts", "checkout_starts", "payments", "completions"]
    labels = ["product view", "add to cart", "checkout start", "payment", "completion"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(stages))
    width = 0.4
    arms = summary["arm"].tolist()
    for i, arm in enumerate(arms):
        row = summary[summary["arm"] == arm].iloc[0]
        n = row["n"]
        values = [row[s] / n * 100 for s in stages]
        ax.barh(y_pos + (i - 0.5) * width, values, height=width, label=arm)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of users (%)")
    ax.set_title("Funnel by arm")
    ax.legend()
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


def power_curve_plot(
    n_grid: np.ndarray,
    power_values: np.ndarray,
    target_power: float = 0.80,
    target_n: int | None = None,
    save_to: Path | None = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n_grid, power_values, linewidth=2)
    ax.axhline(target_power, linestyle="--", linewidth=1, alpha=0.6)
    if target_n is not None:
        ax.axvline(target_n, linestyle="--", linewidth=1, alpha=0.6)
        ax.annotate(
            f"n={target_n:,}",
            xy=(target_n, target_power),
            xytext=(target_n * 1.05, target_power - 0.08),
            fontsize=10,
        )
    ax.set_xlabel("sample size per arm")
    ax.set_ylabel("power")
    ax.set_title("Power vs sample size")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


def ci_forest_plot(
    point_estimates: list[float],
    ci_lows: list[float],
    ci_highs: list[float],
    labels: list[str],
    null_value: float = 0.0,
    xlabel: str = "lift (pp)",
    title: str = "Effect with 95% CI",
    save_to: Path | None = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels) + 2)))
    y = np.arange(len(labels))
    points = np.asarray(point_estimates)
    lows = np.asarray(ci_lows)
    highs = np.asarray(ci_highs)
    ax.errorbar(points, y, xerr=[points - lows, highs - points], fmt="o", capsize=4)
    ax.axvline(null_value, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


def pvalue_histogram(
    pvalues: np.ndarray,
    title: str = "A/A p-value distribution (should be uniform)",
    save_to: Path | None = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pvalues, bins=20, edgecolor="white")
    ax.axhline(len(pvalues) / 20, linestyle="--", linewidth=1, alpha=0.6, label="uniform")
    ax.set_xlabel("p-value")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig

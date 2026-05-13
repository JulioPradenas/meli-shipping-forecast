"""Plotting utilities for consistent visual style across notebooks.

This module configures matplotlib defaults and provides a small palette
designed for high contrast on both light and dark backgrounds.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Color palette: 6 colors chosen for accessibility and print-friendliness.
# Order: primary, secondary, tertiary, success, warning, danger.
PALETTE = [
    "#1f77b4",  # blue (primary)
    "#ff7f0e",  # orange (secondary)
    "#2ca02c",  # green (tertiary / success)
    "#d62728",  # red (danger)
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn defaults for the project."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        rc={
            "figure.figsize": (12, 5),
            "figure.dpi": 100,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    plt.rcParams["font.family"] = "sans-serif"

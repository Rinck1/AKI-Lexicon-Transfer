"""
AKI Research Project — Plotting Utilities
==========================================
Consistent matplotlib/seaborn styling for publication-quality figures.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd

from scripts.utils.constants import OUTPUT_FIGURES

# =============================================================================
# Global Style Configuration
# =============================================================================
# Enable CJK (Chinese) font support — use SimHei on Windows
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
except Exception:
    pass

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

# Color palette — medical/publication friendly
PALETTE = sns.color_palette("Set2", 8)
KDIGO_COLORS = {
    0: "#2ecc71",  # Green — No AKI
    1: "#f1c40f",  # Yellow — Stage 1
    2: "#e67e22",  # Orange — Stage 2
    3: "#e74c3c",  # Red — Stage 3
}
MORTALITY_COLORS = {
    0: "#3498db",  # Blue — Survived
    1: "#e74c3c",  # Red — Died
}

os.makedirs(OUTPUT_FIGURES, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Save figure as both PNG and PDF in the figures output directory.

    Returns the PNG path.
    """
    png_path = os.path.join(OUTPUT_FIGURES, f"{filename}.png")
    pdf_path = os.path.join(OUTPUT_FIGURES, f"{filename}.pdf")
    fig.savefig(png_path, format="png")
    fig.savefig(pdf_path, format="pdf")
    return png_path


def set_title(fig_or_ax, title: str, subtitle: str = ""):
    """Set a publication-style title with optional subtitle."""
    if hasattr(fig_or_ax, "suptitle"):
        fig_or_ax.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        if subtitle:
            fig_or_ax.text(
                0.5, 0.97, subtitle, transform=fig_or_ax.transFigure,
                ha="center", fontsize=10, fontstyle="italic", color="gray",
            )
    else:
        fig_or_ax.set_title(title, fontsize=14, fontweight="bold")
        if subtitle:
            fig_or_ax.text(
                0.5, 1.02, subtitle, transform=fig_or_ax.transAxes,
                ha="center", fontsize=9, fontstyle="italic", color="gray",
            )


def style_axes(ax, xlabel: str = "", ylabel: str = "",
               grid: bool = True, legend: bool = True):
    """Apply consistent styling to an axes."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")
    if legend:
        ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_missing_heatmap(df: pd.DataFrame, filename: str = "missing_heatmap"):
    """Generate missing value heatmap for a DataFrame."""
    fig, ax = plt.subplots(figsize=(16, max(8, len(df.columns) * 0.3)))
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        save_figure(fig, filename)
        return fig
    bars = ax.barh(range(len(missing)), missing.values * 100, color=PALETTE[0])
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index, fontsize=8)
    ax.set_xlabel("Missing (%)")
    ax.invert_yaxis()
    set_title(ax, "Missing Value Distribution", f"{len(missing)} variables with missing data")
    style_axes(ax, legend=False)
    save_figure(fig, filename)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap",
                             n_features: int = 30, target: str = "mortality"):
    """Plot correlation heatmap of top features with mortality."""
    from pandas import DataFrame as Pdf  # noqa — just for type hint
    corr = df.corr()[target].abs().sort_values(ascending=False)
    top = corr.index[1:n_features + 1]  # skip target itself
    corr_matrix = df[list(top) + [target]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7}, ax=ax,
    )
    set_title(ax, f"Top {n_features} Features Correlation with {target}")
    save_figure(fig, filename)
    return fig

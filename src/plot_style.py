"""Unified plotting style for SteamSale notebooks.

Call `apply_style()` once at the top of any notebook (after imports) to get
consistent fonts, colours, figure sizes, and DPI across the project.

Project palette (`PALETTE`) is exported for ad-hoc use in custom plots — keep
to these colours for visual consistency across the report deliverables.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


# Project palette — Steam-inspired with a sale-energy accent
PALETTE = {
    "navy":   "#1B2838",   # dominant dark
    "slate":  "#2A475E",   # dark accent
    "cream":  "#F4F1EA",   # warm off-white
    "ink":    "#1E293B",   # body text
    "muted":  "#64748B",   # captions / hairlines
    "orange": "#E0791C",   # discount-tag accent
    "cyan":   "#66C0F4",   # Steam cyan
    "green":  "#5BA84B",   # positive callouts
    "red":    "#DC2626",   # negative callouts
}

# Ordered tier colours — use everywhere a 4-tier model is plotted so tier-X
# always means the same hue regardless of which notebook produced the chart.
# Both 'High' and 'High Impact' map to the same colour (the data uses the
# verbose form 'High Impact'; some legacy plot legends reference 'High').
TIER_COLORS = {
    "Premium Hold":          PALETTE["green"],
    "Standard Depreciation": PALETTE["cyan"],
    "Heavy Discount":        PALETTE["orange"],
    "Permanent Bargain":     PALETTE["red"],
    # Part 2 — sale-effectiveness tiers
    "High Impact": PALETTE["green"],
    "High":        PALETTE["green"],   # alias for legacy 'High'
    "Moderate":    PALETTE["cyan"],
    "Low":         PALETTE["orange"],
    "None":        PALETTE["red"],
}

# Categorical sequence drawn from PALETTE — sns.set_palette accepts a list.
_CATEGORICAL = [
    PALETTE["navy"], PALETTE["orange"], PALETTE["cyan"],
    PALETTE["green"], PALETTE["slate"], PALETTE["red"],
]


def apply_style() -> None:
    """One-line setup: call once near the top of a notebook."""
    sns.set_theme(
        style="whitegrid",
        palette=_CATEGORICAL,
        rc={
            "figure.dpi":        120,
            "figure.figsize":    (10, 5),
            "axes.titleweight":  "bold",
            "axes.titlesize":    13,
            "axes.labelsize":    11,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.edgecolor":    PALETTE["muted"],
            "grid.color":        "#E5E7EB",
            "grid.linewidth":    0.6,
            "xtick.labelsize":   10,
            "ytick.labelsize":   10,
            "legend.frameon":    False,
            "legend.fontsize":   10,
            "font.family":       "sans-serif",
        },
    )
    # Suppress the pandas/seaborn deprecation chatter that floods notebook
    # output and obscures real warnings.
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

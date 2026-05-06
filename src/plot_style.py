"""Unified plotting style for SteamSale notebooks.

Call `apply_style()` once at the top of any notebook (after imports) to get
consistent fonts, colours, figure sizes, and DPI across the project.

Project palette (`PALETTE`) is exported for ad-hoc use in custom plots — keep
to these colours for visual consistency across the report deliverables.

This is a **dark-mode** palette: purple-primary on a near-black background
with off-white text and grid lines. Designed to drop cleanly into a dark
slide deck. Existing notebook code that references the historical Steam
keys (`cyan`, `orange`, `ink`, `red`, `green`, etc.) keeps working — those
keys are aliased to their dark-mode equivalents below, so no notebook edits
are required.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Project palette
# ---------------------------------------------------------------------------
# Semantic names are the canonical references for new code. Backward-compat
# aliases keep historical notebook code working without churn (they map to
# the closest dark-mode equivalent of the original Steam-themed colour).
PALETTE = {
    # Semantic (recommended for new code)
    "primary":    "#A855F7",   # purple — primary bars/lines
    "secondary":  "#7C3AED",   # deep purple — secondary bars
    "accent":     "#C084FC",   # light purple — highlights
    "positive":   "#34D399",   # green — good / on-target
    "negative":   "#F472B6",   # pink — warning / negative (softer than red)
    "background": "#0D0D1A",   # near-black — figure / axes facecolor
    "grid":       "#2D2B55",   # dark navy-purple — grid lines, hairlines
    "text":       "#E2E8F0",   # off-white — labels, ticks, titles

    # Backward-compat aliases (historical Steam-theme keys still in notebooks)
    # Each maps to the dark-mode equivalent of its original semantic role.
    "cyan":   "#A855F7",   # was Steam cyan (primary chart color)        -> primary purple
    "orange": "#7C3AED",   # was discount-tag accent (secondary)         -> deep purple
    "green":  "#34D399",   # was positive callouts                        -> positive (slight tweak)
    "red":    "#F472B6",   # was negative callouts                        -> warning pink
    "ink":    "#E2E8F0",   # was dark body text on light bg               -> off-white text on dark bg
    "muted":  "#2D2B55",   # was light grey hairlines                     -> grid color
    "navy":   "#0D0D1A",   # was Steam dominant dark                      -> background
    "slate":  "#7C3AED",   # was Steam dark accent                        -> secondary purple
    "cream":  "#C084FC",   # was warm off-white background                -> accent purple
}


# ---------------------------------------------------------------------------
# Tier colours — keep the same semantics (Premium=positive, Bargain=warning)
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Premium Hold":          PALETTE["positive"],
    "Standard Depreciation": PALETTE["primary"],
    "Heavy Discount":        PALETTE["secondary"],
    "Permanent Bargain":     PALETTE["negative"],
    # Part 2 — sale-effectiveness tiers
    "High Impact": PALETTE["positive"],
    "High":        PALETTE["positive"],   # alias for legacy 'High'
    "Moderate":    PALETTE["primary"],
    "Low":         PALETTE["secondary"],
    "None":        PALETTE["negative"],
}


# Categorical sequence drawn from PALETTE — sns.set_palette accepts a list.
# Order matters: the first colour is the default for line/bar plots.
_CATEGORICAL = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["positive"],
    PALETTE["accent"],
    PALETTE["negative"],
    PALETTE["text"],
]


# ---------------------------------------------------------------------------
# Custom dark-friendly colormap for heatmaps (replaces matplotlib's 'Blues',
# which has a near-white low end that disappears against a dark background).
# Sequential dark-bg-friendly: background -> grid -> secondary -> accent.
# ---------------------------------------------------------------------------
PURPLE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "steam_purple",
    [PALETTE["background"], PALETTE["grid"], PALETTE["secondary"], PALETTE["accent"]],
)


def apply_style() -> None:
    """One-line setup: call once near the top of a notebook.

    Switches matplotlib + seaborn to a dark-mode aesthetic that matches the
    project palette. Sets figure / axes / text / grid colours so charts drop
    cleanly into a dark slide deck.
    """
    sns.set_theme(
        style="darkgrid",
        palette=_CATEGORICAL,
        rc={
            # Layout / typography
            "figure.dpi":          120,
            "figure.figsize":      (10, 5),
            "axes.titleweight":    "bold",
            "axes.titlesize":      13,
            "axes.labelsize":      11,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "xtick.labelsize":     10,
            "ytick.labelsize":     10,
            "legend.frameon":      False,
            "legend.fontsize":     10,
            "font.family":         "sans-serif",

            # Dark theme — backgrounds
            "figure.facecolor":    PALETTE["background"],
            "figure.edgecolor":    PALETTE["background"],
            "axes.facecolor":      PALETTE["background"],
            "savefig.facecolor":   PALETTE["background"],
            "savefig.edgecolor":   PALETTE["background"],

            # Dark theme — text + ticks (all routed to off-white)
            "text.color":          PALETTE["text"],
            "axes.labelcolor":     PALETTE["text"],
            "axes.titlecolor":     PALETTE["text"],
            "axes.edgecolor":      PALETTE["text"],
            "xtick.color":         PALETTE["text"],
            "ytick.color":         PALETTE["text"],
            "legend.labelcolor":   PALETTE["text"],

            # Grid lines — dark navy-purple (visible on dark bg without dominating)
            "grid.color":          PALETTE["grid"],
            "grid.linewidth":      0.6,
            "grid.alpha":          0.55,

            # Default colormap for imshow / pcolormesh — replaces 'viridis'
            # default with our project purple ramp. Notebooks that explicitly
            # set cmap=... override this; this only affects plots that don't.
            "image.cmap":          "steam_purple",
        },
    )
    # Register our cmap with matplotlib so cmap='steam_purple' works in any
    # call (including the rcParams default above).
    try:
        plt.colormaps.register(cmap=PURPLE_CMAP, force=True)
    except (ValueError, AttributeError):
        # Older matplotlib (<3.5): fall back to register_cmap
        try:
            plt.register_cmap(name="steam_purple", cmap=PURPLE_CMAP)
        except Exception:
            pass

    # Suppress the pandas/seaborn deprecation chatter that floods notebook
    # output and obscures real warnings.
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

"""Generate docs/db_schema.png — an ER-style diagram of the SteamSale DB.

Re-run this whenever the schema changes:
    python scripts/make_db_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "db_schema.png"

# Color palette grouped by data domain
DOMAIN = {
    "worklist": ("#4b5563", "#e5e7eb"),  # gray
    "steam":    ("#0f4c81", "#dbeafe"),  # storefront blue
    "reviews":  ("#1d4ed8", "#dbeafe"),  # blue
    "steamspy": ("#15803d", "#dcfce7"),  # green
    "itad":     ("#c2410c", "#ffedd5"),  # orange
    "players":  ("#6d28d9", "#ede9fe"),  # purple
}

# (name, [columns], domain, center_x, center_y)
TABLES = [
    ("app_list", [
        "appid  PK",
        "name", "source", "added_at",
        "has_* progress flags (6)",
        "last_error",
    ], "worklist", 6.0, 8.2),

    ("games", [
        "appid  PK, FK→app_list",
        "title, type, is_free",
        "developer, publisher",
        "release_date, currency",
        "launch_price_cents",
        "current_price_cents",
        "discount_percent",
        "metacritic_score",
        "windows / mac / linux",
        "achievements_total",
        "controller_support",
        "fetched_at, raw_json",
    ], "steam", 6.0, 5.0),

    ("game_genres", ["appid  FK", "genre"], "steam", 1.2, 1.4),
    ("game_categories", ["appid  FK", "category"], "steam", 1.2, 0.0),

    ("reviews_summary", [
        "appid  PK,FK", "total_reviews",
        "total_positive", "total_negative",
        "review_score", "review_score_desc",
    ], "reviews", 3.6, 1.4),
    ("review_timestamps", [
        "review_id  PK", "appid  FK",
        "timestamp_created", "voted_up",
    ], "reviews", 3.6, -0.3),

    ("steamspy", [
        "appid  PK,FK", "owners_min", "owners_max",
        "average_forever", "average_2weeks",
        "median_forever", "median_2weeks",
        "ccu", "score_rank",
    ], "steamspy", 6.0, 1.4),
    ("steamspy_tags", [
        "appid  FK", "tag", "votes",
    ], "steamspy", 6.0, -0.7),

    ("itad_mapping", [
        "appid  PK,FK", "itad_id", "slug", "fetched_at",
    ], "itad", 8.4, 1.4),
    ("price_history", [
        "appid  FK", "shop_id", "shop_name",
        "timestamp", "price_amount", "regular_amount",
        "cut", "deal_id",
    ], "itad", 8.4, -0.6),

    ("player_counts", [
        "appid  FK", "fetched_at", "player_count",
    ], "players", 10.8, 1.4),
    ("steamcharts_history", [
        "appid  FK", "timestamp", "player_count",
    ], "players", 10.8, -0.4),
]

# (child, parent) foreign keys
FKS = [
    ("games", "app_list"),
    ("game_genres", "games"),
    ("game_categories", "games"),
    ("reviews_summary", "games"),
    ("review_timestamps", "games"),
    ("steamspy", "games"),
    ("steamspy_tags", "games"),
    ("itad_mapping", "games"),
    ("price_history", "games"),
    ("player_counts", "games"),
    ("steamcharts_history", "games"),
]

# Layout constants
BOX_W = 2.05
HEADER_H = 0.28
ROW_H = 0.18
PADDING = 0.10


def draw_table(ax, name, cols, domain, cx, cy):
    border, fill = DOMAIN[domain]
    rows = len(cols)
    h = HEADER_H + rows * ROW_H + PADDING

    x0 = cx - BOX_W / 2
    y0 = cy - h / 2

    # Body
    body = FancyBboxPatch(
        (x0, y0), BOX_W, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1, edgecolor=border, facecolor="white",
        zorder=2,
    )
    ax.add_patch(body)

    # Header strip
    header = FancyBboxPatch(
        (x0, y0 + h - HEADER_H), BOX_W, HEADER_H,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0, facecolor=border, zorder=3,
    )
    ax.add_patch(header)
    ax.text(
        cx, y0 + h - HEADER_H / 2, name,
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=4,
    )

    # Tint zone behind columns
    tint = FancyBboxPatch(
        (x0, y0), BOX_W, h - HEADER_H,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0, facecolor=fill, zorder=2.5,
    )
    ax.add_patch(tint)

    # Columns
    for i, col in enumerate(cols):
        y = y0 + h - HEADER_H - (i + 0.5) * ROW_H - PADDING / 2
        ax.text(
            x0 + 0.08, y, col,
            ha="left", va="center", fontsize=7.2, color="#111827",
            zorder=4,
        )

    return {"x0": x0, "y0": y0, "x1": x0 + BOX_W, "y1": y0 + h, "cx": cx, "cy": cy}


def edge_point(rect, target_cx, target_cy):
    """Return the point on `rect`'s border closest to (target_cx, target_cy)."""
    cx, cy = rect["cx"], rect["cy"]
    dx = target_cx - cx
    dy = target_cy - cy
    if dx == 0 and dy == 0:
        return cx, cy

    # Half-extents
    hx = (rect["x1"] - rect["x0"]) / 2
    hy = (rect["y1"] - rect["y0"]) / 2

    # Scale so that the point lands on whichever edge is hit first.
    if abs(dx) * hy > abs(dy) * hx:
        # Hits left/right edge
        s = hx / abs(dx)
    else:
        s = hy / abs(dy)
    return cx + dx * s, cy + dy * s


def main():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(-1.7, 9.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        6.0, 9.5, "SteamSale — Database Schema",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#111827",
    )
    ax.text(
        6.0, 9.15,
        "12 tables: 1 worklist  +  1 master  +  10 child tables (FK → games)",
        ha="center", va="center", fontsize=9, color="#4b5563",
    )

    rects = {}
    for name, cols, domain, cx, cy in TABLES:
        rects[name] = draw_table(ax, name, cols, domain, cx, cy)

    # Foreign-key arrows
    for child, parent in FKS:
        cr = rects[child]
        pr = rects[parent]
        x1, y1 = edge_point(cr, pr["cx"], pr["cy"])
        x2, y2 = edge_point(pr, cr["cx"], cr["cy"])
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=0.9, color="#6b7280", zorder=1,
        )
        ax.add_patch(arrow)

    # Legend
    legend_x = 0.0
    legend_y = -1.4
    legend_items = [
        ("worklist", "Resumable worklist"),
        ("steam",    "Steam Storefront"),
        ("reviews",  "Steam Reviews"),
        ("steamspy", "SteamSpy"),
        ("itad",     "IsThereAnyDeal"),
        ("players",  "Player counts (live + historical)"),
    ]
    for i, (key, label) in enumerate(legend_items):
        border, fill = DOMAIN[key]
        x = legend_x + (i % 3) * 4.2
        y = legend_y - (i // 3) * 0.32
        swatch = FancyBboxPatch(
            (x, y - 0.08), 0.28, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.8, edgecolor=border, facecolor=fill, zorder=2,
        )
        ax.add_patch(swatch)
        ax.text(x + 0.38, y, label, ha="left", va="center", fontsize=8.2, color="#1f2937")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()

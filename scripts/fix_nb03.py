"""Apply the audit fixes to notebook 03.

Edits:
  - cell e62683a2 (heavy methodology markdown):
      Strip the stale REMOVED-features rationale (developer_tier and
      value_retention_tier are now actually gone from the data, not just
      "not used"). Replace with a tighter feature-set summary.
  - cell pca_prep:
      Update the trailing REMOVED block; add log_initial_price,
      log_total_reviews, log_achievements_total to numeric_features.
  - cell target_regression:
      Switch target from current_price (PHP) to max_discount_ever (matches
      notebook 04 regression). Replace with a clean USD-aware visualization.
  - cell target_classification:
      Replace value_retention_tier analysis with quartile bins on
      max_discount_ever (data-derived, equal-mass).
  - cell pca_viz_2d:
      Color by max_discount_ever instead of value_retention_tier; switch
      price colorbar from PHP to USD.
  - cell corr_target:
      Correlate against max_discount_ever instead of current_price; replace
      box-plot panel with quartile-based bins.

New cells inserted:
  - target_max_discount_dist_md + cell after the filter cell with the actual
    target distribution + has_itad_data summary.

Compile-checks every replacement and adds a final ascii smoke check that no
broken-column references survive.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB   = ROOT / "notebooks" / "03_eda.ipynb"
SRC  = ROOT / "scripts"

nb = json.loads(NB.read_text(encoding="utf-8"))
get_cell = lambda cid: next((c for c in nb["cells"] if c.get("id") == cid), None)

def set_source(cell, src_str, *, parse=True):
    if parse:
        ast.parse(src_str)
    cell["source"]          = src_str.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None


# ---- 1. Methodology markdown (e62683a2) ---------------------------------
methodology_md = [
    "## Methodology Notes\n",
    "\n",
    "**Feature set in this notebook mirrors the modeling notebooks (04, 04c, 05).**\n",
    "\n",
    "### Continuous numeric features (8)\n",
    "- `days_since_release`, `release_year`, `release_month`\n",
    "- `initial_price` (USD, ITAD-derived) + `log_initial_price`\n",
    "- `review_score` (positive ratio)\n",
    "- `log_total_reviews` (raw `total_reviews` is heavy-tailed: median 28, max 9.5M)\n",
    "- `log_ownership` (SteamSpy band midpoint)\n",
    "- `log_achievements_total` (raw is capped at 5,000 by Steam's API for achievement-farm games)\n",
    "\n",
    "### Categorical features (4)\n",
    "- `primary_genre` (Steam's first listed genre)\n",
    "- `is_multiplayer`, `has_controller_support` (binary, derived from Steam categories)\n",
    "- `player_engagement` (Low / Medium / High by 100, 1000 player thresholds)\n",
    "\n",
    "### High-cardinality (1)\n",
    "- `publisher` -- target-encoded in modeling, here just descriptive\n",
    "\n",
    "### Target\n",
    "- **Continuous**: `max_discount_ever / 100` -- 0 to 1, the deepest discount each game has reached. This is what notebook 04 Step 4.4 regresses.\n",
    "- **Discrete (for narrative only)**: quartile bins of the same target -- shown later in this notebook for readability, but not the actual model target.\n",
    "\n",
    "### Removed during the cleaning audit (gone from `cleaned_games`):\n",
    "- `developer_tier` -- the AAA hand-list had ~18% substring false positives, and the Indie branch was unreachable (SteamSpy floor was 200k owners). Replaced by `publisher` target-encoding in the model.\n",
    "- `value_retention_tier` -- the 0.85 / 0.50 / 0.25 boundaries were hand-picked; the `Premium Hold` definition (`ever_discounted == 0`) conflicted with the modeling filter that requires `max_discount_ever > 0`. Replaced by direct discount-depth modeling.\n",
    "- `price_tier` (kept on `cleaned_games` for narrative) -- model uses continuous `initial_price` instead.\n",
    "- `discount_depth` (game-level, single-snapshot) -- redundant with `max_discount_ever`.\n",
    "- `owners` and `price_retention` -- aliases of `ownership_midpoint` and `price_retention_ratio`; dropped to declutter.\n",
    "\n",
    "### Notebook 04 changes reflected here\n",
    "- Log-transformed numerics persist on `cleaned_games`, so EDA reads them directly.\n",
    "- Filter funnel matches Step 4.2 exactly (publisher, release_month, primary_genre, etc. must all be non-null).\n",
]
get_cell("e62683a2")["source"] = methodology_md
print("cell e62683a2: methodology markdown rewritten (stale REMOVED rationale dropped)")


# ---- 2. pca_prep -- add log columns + clean up REMOVED block -------------
cell = get_cell("pca_prep")
src = "".join(cell["source"])

# Remove the existing log_total_reviews derivation since it's now in cleaned_games
old_derive = "# Create log_total_reviews\ndf['log_total_reviews'] = np.log10(df['total_reviews'] + 1)\n"
new_derive = ("# Defensive backfills -- log columns are persisted by the audit-fixed\n"
              "# notebook 02, but compute them here too so this notebook works even\n"
              "# against an older `cleaned_games` snapshot.\n"
              "if 'log_total_reviews' not in df.columns:\n"
              "    df['log_total_reviews'] = np.log1p(df['total_reviews'].fillna(0))\n"
              "if 'log_initial_price' not in df.columns:\n"
              "    df['log_initial_price'] = np.log1p(df['initial_price'].fillna(0))\n"
              "if 'log_achievements_total' not in df.columns:\n"
              "    df['log_achievements_total'] = np.log1p(df['achievements_total'].fillna(0))\n")
if old_derive in src:
    src = src.replace(old_derive, new_derive, 1)

# Update numeric_features list
old_num = """numeric_features = [
    'days_since_release',
    'initial_price',      # Continuous (not binned into price_tier)
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'achievements_total',
    'avg_recent_players'  # Continuous version (for regression)
]"""
new_num = """numeric_features = [
    'days_since_release',
    'initial_price',          # raw USD
    'log_initial_price',      # log1p, matches Step 4.2's NUM_FEATURES
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'log_achievements_total', # log1p, matches Step 4.2's NUM_FEATURES
    'avg_recent_players',     # continuous live-player signal
]"""
if old_num in src:
    src = src.replace(old_num, new_num, 1)

# Replace the trailing REMOVED block
old_removed_marker = "?? REMOVED (no research backing):"
if old_removed_marker in src:
    head, tail = src.split(old_removed_marker, 1)
    # Drop everything from the REMOVED marker to the end of the cell;
    # close the print block cleanly.
    src = head.rstrip() + (
        '\n\nprint("\\n[OK] Feature set finalized -- matches Step 4.2 NUM/CAT/PUB lists.")\n'
    )

ast.parse(src)
set_source(cell, src, parse=False)
print("cell pca_prep: log columns added; stale REMOVED block dropped")


# ---- 3. target_regression -- switch from current_price to max_discount_ever
target_regression_src = (SRC / "_nb03_target_regression.py").read_text(encoding="utf-8")
set_source(get_cell("target_regression"), target_regression_src)
print("cell target_regression: target switched to max_discount_ever; PHP -> USD")


# ---- 4. target_classification -- quartile bins on max_discount_ever ------
target_cls_src = (SRC / "_nb03_target_classification.py").read_text(encoding="utf-8")
set_source(get_cell("target_classification"), target_cls_src)
print("cell target_classification: rebuilt as quartile bins on max_discount_ever")


# ---- 5. pca_viz_2d -- color by max_discount_ever -------------------------
pca_viz_src = (SRC / "_nb03_pca_viz_2d.py").read_text(encoding="utf-8")
set_source(get_cell("pca_viz_2d"), pca_viz_src)
print("cell pca_viz_2d: coloring switched to max_discount_ever; price labeled USD")


# ---- 6. corr_target -- correlate vs max_discount_ever --------------------
corr_target_src = (SRC / "_nb03_corr_target.py").read_text(encoding="utf-8")
set_source(get_cell("corr_target"), corr_target_src)
print("cell corr_target: correlation target switched to max_discount_ever")


# ---- 7. summary_output -- update the wording ------------------------------
cell = get_cell("summary_output")
src = "".join(cell["source"])
# Two simple text swaps: PHP -> USD; current_price target -> max_discount_ever
src = src.replace("Current Price (PHP)", "max_discount_ever (USD-derived)")
src = src.replace("'current_price'", "'max_discount_ever'")
src = src.replace("? ?", "")
ast.parse(src)
set_source(cell, src, parse=False)
print("cell summary_output: PHP -> USD; current_price -> max_discount_ever")


# ---- 8. Insert a new cell: target distribution headline + has_itad_data ---
target_section_md_id = "target_md_new"
target_section_code_id = "target_code_new"

# Don't double-insert if rerun
if not any(c.get("id") == target_section_code_id for c in nb["cells"]):
    md = {
        "cell_type": "markdown",
        "id": target_section_md_id,
        "metadata": {},
        "source": [
            "## Headline target view\n",
            "\n",
            "Quick stats on the regression target (`max_discount_ever`) plus ITAD coverage. Useful before the deeper regression-distribution panels below.\n",
        ],
    }
    code_src = '''target = df['max_discount_ever'] / 100
itad_covered = df['has_itad_data'].sum() if 'has_itad_data' in df.columns else df['max_discount_ever'].notna().sum()

print("HEADLINE TARGET VIEW")
print("-" * 60)
print(f"Games in modeling dataset            : {len(df):>5,}")
print(f"With ITAD price-history coverage     : {itad_covered:>5,}  ({itad_covered/len(df)*100:.1f}%)")
print()
print(f"max_discount_ever / 100 (regression target):")
print(f"  mean   : {target.mean():.3f}")
print(f"  median : {target.median():.3f}")
print(f"  std    : {target.std():.3f}")
print(f"  share >= 50% off ever: {(target >= 0.5).mean()*100:.1f}%")
print(f"  share >= 75% off ever: {(target >= 0.75).mean()*100:.1f}%")
'''
    ast.parse(code_src)
    code = {
        "cell_type": "code",
        "id": target_section_code_id,
        "metadata": {},
        "source": code_src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }
    # Insert right before the existing target_section markdown
    target_idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "target_section")
    nb["cells"] = nb["cells"][:target_idx] + [md, code] + nb["cells"][target_idx:]
    print(f"inserted new cells {target_section_md_id} + {target_section_code_id}")


# ---- Save ---
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\n{NB.name} saved -- {len(nb['cells'])} cells.")

# ---- Smoke check: confirm no broken references survive in code cells ---
nb_check = json.loads(NB.read_text(encoding="utf-8"))
src_full = "\n".join("".join(c["source"]) for c in nb_check["cells"] if c["cell_type"] == "code")
broken = []
for needle in ("developer_tier", "value_retention_tier", "df['owners']", "df['price_retention']"):
    if needle in src_full:
        broken.append(needle)
if broken:
    print("WARNING: still referenced in code cells:", broken)
else:
    print("OK: no broken-column references survive in code cells.")

"""Apply nb 04's model-equivalent changes to nb 05.

Nb 05 doesn't reference any of the columns we removed (developer_tier,
value_retention_tier, price_retention, owners), so it still runs. The only
nb-04-equivalent improvement applicable here is the LOG TRANSFORMS for
heavy-tailed numerics (achievements_total, total_reviews).

Two surgical edits:
  1. cell e5f6a7b8 (feature engineering) -- derive log_total_reviews and
     log_achievements_total defensively (in case cleaned_sale_events is
     stale).
  2. cell f6a7b8c9 (BASE_FEATURES list) -- swap raw -> log versions,
     add log_total_reviews.

Note: we do NOT drop price_tier here. In nb 04 it was redundant with
initial_price (both encoded the same signal). In nb 05 there is no
initial_price column at all, so price_tier is the only price signal.
Dropping it would lose information; we keep it.

Also: we do NOT add publisher target encoding here. The flat-numpy-array
pipeline in nb 05 would need substantial refactoring (manual K-fold target
encoder to avoid leakage) to add it cleanly. That's an enhancement, not a
parity fix; flagged as a follow-up.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB   = ROOT / "notebooks" / "05_ml_part2_sale_effectiveness.ipynb"
nb   = json.loads(NB.read_text(encoding="utf-8"))


def get_cell(cid):
    return next(c for c in nb["cells"] if c.get("id") == cid)


# ---- 1. cell e5f6a7b8 -- add log derivations -----------------------------
cell = get_cell("e5f6a7b8")
src  = "".join(cell["source"])

# Add the log-transform block right before the discount_depth fillna line so it
# sits with the other game-level numeric prep.
old_anchor = """df['cumulative_sale_count']= df['cumulative_sale_count'].fillna(0)

# discount_depth from cleaned_sale_events is the depth of THIS sale event (0-1)"""
new_block = """df['cumulative_sale_count']= df['cumulative_sale_count'].fillna(0)

# Log-transforms for heavy-tailed game-level numerics (matches nb 04's feature
# space). Derived defensively so this still works if cleaned_sale_events
# pre-dates the notebook 02 audit fixes.
df['log_total_reviews']      = np.log1p(df.get('total_reviews', pd.Series(0, index=df.index)).fillna(0))
df['log_achievements_total'] = np.log1p(df['achievements_total'].fillna(0))

# discount_depth from cleaned_sale_events is the depth of THIS sale event (0-1)"""

if old_anchor not in src:
    raise SystemExit("cell e5f6a7b8: anchor for log block not found verbatim")
src = src.replace(old_anchor, new_block, 1)

# Drop the now-redundant achievements_total fillna at the end (we already
# coalesce in log1p). Also leave the raw column in df so nothing else breaks.
old_tail = """df['is_multiplayer']         = df['is_multiplayer'].fillna(0).astype(int)
df['has_controller_support'] = df['has_controller_support'].fillna(0).astype(int)
df['achievements_total']     = df['achievements_total'].fillna(0)"""
new_tail = """df['is_multiplayer']         = df['is_multiplayer'].fillna(0).astype(int)
df['has_controller_support'] = df['has_controller_support'].fillna(0).astype(int)
df['achievements_total']     = df['achievements_total'].fillna(0)   # kept raw too in case any downstream cell wants it"""
if old_tail in src:
    src = src.replace(old_tail, new_tail, 1)

ast.parse(src)
cell["source"]          = src.splitlines(keepends=True)
cell["outputs"]         = []
cell["execution_count"] = None
print("cell e5f6a7b8: added log_total_reviews + log_achievements_total derivations")


# ---- 2. cell f6a7b8c9 -- update BASE_FEATURES ----------------------------
cell = get_cell("f6a7b8c9")
src  = "".join(cell["source"])

old_features = """BASE_FEATURES = [
    # Game attributes (game-level, repeated per sale event)
    'age_at_sale_days', 'review_score_pct', 'log_ownership',
    'is_multiplayer', 'has_controller_support', 'achievements_total',
    # Sale attributes (event-level)
    'discount_depth', 'days_since_last_sale', 'cumulative_sale_count',
    # Interaction
    'depth_x_age',
]"""
new_features = """BASE_FEATURES = [
    # Game attributes (game-level, repeated per sale event).
    # achievements_total + total_reviews are heavy-tailed (log-transformed
    # to match nb 04; raw values would let the rare 5000-achievement farms
    # dominate StandardScaler's distribution).
    'age_at_sale_days', 'review_score_pct', 'log_ownership',
    'is_multiplayer', 'has_controller_support',
    'log_achievements_total', 'log_total_reviews',
    # Sale attributes (event-level)
    'discount_depth', 'days_since_last_sale', 'cumulative_sale_count',
    # Interaction
    'depth_x_age',
]"""
if old_features not in src:
    raise SystemExit("cell f6a7b8c9: BASE_FEATURES anchor not found verbatim")
src = src.replace(old_features, new_features, 1)

ast.parse(src)
cell["source"]          = src.splitlines(keepends=True)
cell["outputs"]         = []
cell["execution_count"] = None
print("cell f6a7b8c9: BASE_FEATURES uses log_achievements_total + log_total_reviews")


NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\n{NB.name} updated.")

"""Apply audit fixes to notebook 02, notebook 04, and the database.

Notebook 02:
  - cell b3c7c01f: drop rows with unparseable release_date instead of
                   silently filling with -1 (was producing wrong is_legacy/age tier)
  - cell da8dea8c: add `has_itad_data` flag, drop the stale `discount_depth`
                   column (single-snapshot, redundant with `max_discount_ever`)
  - cell bfea13a0: add log_total_reviews, log_initial_price, log_achievements_total
                   so log columns persist in cleaned_games (currently re-derived
                   in notebook 04 only)
  - cell 401d526e: remove developer_tier (Indie branch dead, AAA list buggy)
  - cell 4e2b9530: simplify Boxleiter to flat multiplier of 40 (no dev_tier)
  - cell ecd821cb: simplify to price_retention_ratio (NaN for invalid),
                   remove value_retention_tier + price_retention alias
  - cell da02b8a4: update cols_part1 -- drop dropped fields, add new ones
  - cell 73414044: remove developer_tier + owners from game_feats merge
  - cell e218a807: drop developer_tier + owners from cols_part2
  - cell 1cf27364: drop developer_tier from panel feature merge

Notebook 04:
  - cell 02ef2e3c: remove dominant_dev pull (column no longer exists)

Database:
  - DROP TABLE player_counts (empty, deprecated)

src/db.py:
  - Remove player_counts schema
"""
import ast
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB02 = ROOT / "notebooks" / "02_data_cleaning.ipynb"
NB04 = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
DB   = ROOT / "data"      / "steam.db"
SRC_DB = ROOT / "src"     / "db.py"


def get_cell(nb, cid):
    c = next((c for c in nb["cells"] if c.get("id") == cid), None)
    if c is None:
        raise SystemExit(f"cell {cid} not found")
    return c


def set_source(cell, src_str):
    """Replace the cell's source with src_str, parse-checking first."""
    ast.parse(src_str)
    cell["source"]          = src_str.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None


# =============================================================================
# Notebook 02 patches
# =============================================================================
nb = json.loads(NB02.read_text(encoding="utf-8"))

# ---- b3c7c01f: drop unparseable dates instead of fillna(-1) ----
new_age = '''# Age features (TIMEZONE-AWARE)
def parse_release_date(date_str):
    if pd.isna(date_str) or date_str == '':
        return pd.NaT
    for fmt in ['%b %d, %Y', '%d %b, %Y', '%Y-%m-%d', '%B %d, %Y']:
        try:
            return pd.Timestamp(datetime.strptime(date_str, fmt))
        except:
            continue
    return pd.NaT

df['release_date_parsed'] = df['release_date'].apply(parse_release_date)
df['release_date_parsed'] = pd.to_datetime(df['release_date_parsed'], utc=True)

# Drop games with unparseable release dates -- we cannot compute age features
# for them, and the previous fillna(-1) was producing silently-wrong derived
# values (is_legacy=0, value_retention_tier=Too New, etc.).
n_before = len(df)
df = df[df['release_date_parsed'].notna()].copy()
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"  Dropped {n_dropped:,} games with unparseable release dates")

now_utc = pd.Timestamp.now(tz='UTC')
df['days_since_release'] = (now_utc - df['release_date_parsed']).dt.days.astype(int)
df['release_year']       = df['release_date_parsed'].dt.year.astype(int)
df['release_month']      = df['release_date_parsed'].dt.month.astype(int)
df['is_legacy']          = (df['days_since_release'] > 1825).astype(int)
print(f"[OK] Age features: {len(df):,} games with valid dates")
'''
set_source(get_cell(nb, "b3c7c01f"), new_age)
print("cell b3c7c01f: drops unparseable-date rows")

# ---- da8dea8c: add has_itad_data, drop discount_depth derivation ----
cell = get_cell(nb, "da8dea8c")
src = "".join(cell["source"])

# Add has_itad_data right after the latest_snapshot merge.
old_after_merge = """df = df.merge(latest_snapshot, on='appid', how='left')
df = df.merge(price_stats,     on='appid', how='left')
df['has_dynamic_pricing'] = df['min_historical_price'].notna().astype(int)"""
new_after_merge = """df = df.merge(latest_snapshot, on='appid', how='left')
df = df.merge(price_stats,     on='appid', how='left')
df['has_dynamic_pricing'] = df['min_historical_price'].notna().astype(int)

# `has_itad_data` is the cleaner version of has_dynamic_pricing -- 1 if ITAD
# returned at least one price-history row for this game, 0 otherwise. Used
# downstream to filter games whose 'never discounted' label is a data artifact.
df['has_itad_data'] = df['initial_price'].notna().astype(int)"""
if old_after_merge not in src:
    raise SystemExit("da8dea8c: latest_snapshot merge anchor not found")
src = src.replace(old_after_merge, new_after_merge, 1)

# Drop the stale single-snapshot discount_depth derivation.
old_disc_depth = """# 3. Discount depth from current snapshot's cut% (currency-free, exact)
#    Equivalent to (initial_price - current_price) / initial_price; using
#    cut directly avoids float drift.
df['discount_depth'] = (df['current_cut'] / 100).fillna(0).clip(0, 1)

"""
if old_disc_depth not in src:
    raise SystemExit("da8dea8c: discount_depth block anchor not found")
src = src.replace(old_disc_depth, "", 1)

set_source(cell, src)
print("cell da8dea8c: added has_itad_data, removed stale discount_depth")

# ---- bfea13a0: add log_total_reviews, log_initial_price, log_achievements_total ----
cell = get_cell(nb, "bfea13a0")
src = "".join(cell["source"])
old_recep = """df['ownership_midpoint'] = (df['owners_min'] + df['owners_max']) / 2
df['owners'] = df['ownership_midpoint']  # Alias
df['log_ownership'] = np.log10(df['ownership_midpoint'] + 1)
print(\"? Reception features created\")"""
new_recep = """df['ownership_midpoint'] = (df['owners_min'] + df['owners_max']) / 2
df['log_ownership'] = np.log10(df['ownership_midpoint'] + 1)

# Log-transforms for heavy-tailed numerics (persist here so notebook 03 / 04 / 06
# don't have to re-derive them inline). log1p handles zeros cleanly.
df['log_total_reviews']      = np.log1p(df['total_reviews'])
df['log_initial_price']      = np.log1p(df['initial_price'].fillna(0))
df['log_achievements_total'] = np.log1p(df['achievements_total'])
print(\"[OK] Reception features + log columns created\")"""
if old_recep not in src:
    # Maybe the question-mark glyph differs by encoding -- try a more lenient match
    if "df['ownership_midpoint'] = (df['owners_min'] + df['owners_max']) / 2" not in src:
        raise SystemExit("bfea13a0: reception block anchor not found")
    # Replace just the lines we know about
    src = src.replace(
        "df['owners'] = df['ownership_midpoint']  # Alias\n",
        "",
        1,
    )
    src = src.replace(
        "df['log_ownership'] = np.log10(df['ownership_midpoint'] + 1)",
        "df['log_ownership'] = np.log10(df['ownership_midpoint'] + 1)\n\n"
        "# Log-transforms for heavy-tailed numerics (persist here so notebook 03 / 04 / 06\n"
        "# don't have to re-derive them inline). log1p handles zeros cleanly.\n"
        "df['log_total_reviews']      = np.log1p(df['total_reviews'])\n"
        "df['log_initial_price']      = np.log1p(df['initial_price'].fillna(0))\n"
        "df['log_achievements_total'] = np.log1p(df['achievements_total'])",
        1,
    )
else:
    src = src.replace(old_recep, new_recep, 1)
set_source(cell, src)
print("cell bfea13a0: added log_total_reviews, log_initial_price, log_achievements_total + dropped owners alias")

# ---- 401d526e: remove developer_tier ----
cell = get_cell(nb, "401d526e")
src = "".join(cell["source"])
old_devtier = """# Developer tier
AAA = ['Valve', 'EA', 'Ubisoft', 'Activision', 'Rockstar', 'Bethesda', 'Square Enix', 'Capcom']
df['developer_tier'] = df.apply(lambda r: 'AAA' if any(a in str(r['developer']) for a in AAA) else ('Indie' if r['ownership_midpoint'] < 50000 else 'Mid-tier'), axis=1)

"""
if old_devtier not in src:
    raise SystemExit("401d526e: developer_tier block anchor not found")
src = src.replace(old_devtier, "", 1)
set_source(cell, src)
print("cell 401d526e: developer_tier derivation removed")

# ---- 4e2b9530: flat Boxleiter multiplier ----
new_box = '''# Boxleiter unit-sales estimate.
# Original used a per-tier multiplier (AAA=30, Mid=65, Indie=40) but our
# AAA hand-list had ~18% substring false positives, and the Indie branch
# was unreachable (SteamSpy floor was 200k owners). We replace it with a
# flat indie-baseline multiplier of 40, which is conservative and avoids
# the bogus tier discrimination. units_sold_estimate isn't a model feature
# anyway -- it's used for EDA narrative only.
df['boxleiter_multiplier'] = 40
df['units_sold_estimate']  = (df['total_reviews'] * df['boxleiter_multiplier']).fillna(0).astype(int)
print("[OK] Boxleiter calculated (flat multiplier=40)")
'''
set_source(get_cell(nb, "4e2b9530"), new_box)
print("cell 4e2b9530: Boxleiter simplified to flat multiplier")

# ---- ecd821cb: simplify to price_retention_ratio (NaN for invalid) ----
new_target = '''# Price retention ratio: current_price / initial_price.
# Only defined when initial_price > 0 AND game is at least 1 year old AND
# current_price is observed -- otherwise NaN (not 0!), so downstream filters
# can distinguish 'genuinely deeply discounted' from 'no valid ratio'.
df['price_retention_ratio'] = np.where(
    (df['initial_price'] > 0)
    & (df['days_since_release'] >= 365)
    & df['current_price'].notna(),
    df['current_price'] / df['initial_price'],
    np.nan,
)
print("[OK] price_retention_ratio computed (NaN where undefined)")
'''
set_source(get_cell(nb, "ecd821cb"), new_target)
print("cell ecd821cb: simplified target derivation, dropped value_retention_tier")

# ---- da02b8a4: update cols_part1 ----
new_save = '''# Save Part 1
cols_part1 = [
    'appid', 'title', 'developer', 'publisher', 'release_date',
    'days_since_release', 'release_year', 'release_month', 'is_legacy',
    'initial_price', 'log_initial_price', 'current_price',
    'price_tier', 'price_retention_ratio',
    'review_score', 'total_reviews', 'log_total_reviews',
    'total_positive', 'total_negative',
    'ownership_midpoint', 'log_ownership', 'units_sold_estimate',
    'primary_genre', 'is_multiplayer', 'has_controller_support',
    'achievements_total', 'log_achievements_total',
    'ever_discounted', 'max_discount_ever',
    'player_engagement', 'avg_recent_players',
    'has_itad_data',
]

df_part1 = df[cols_part1].copy()
df_part1.to_csv('part1_game_level_cleaned.csv', index=False)
print(f"\\n[OK] PART 1 SAVED: {len(df_part1):,} games, {len(df_part1.columns)} columns")
'''
set_source(get_cell(nb, "da02b8a4"), new_save)
print("cell da02b8a4: cols_part1 updated (+log columns +has_itad_data, -dev_tier, -value_retention_tier, -aliases, -discount_depth)")

# ---- 73414044: remove developer_tier + owners alias from game_feats ----
cell = get_cell(nb, "73414044")
src = "".join(cell["source"])
old_gfeats = """game_feats = [
    'appid', 'title', 'developer',
    'primary_genre', 'developer_tier', 'price_tier',
    'review_score', 'total_reviews',
    'ownership_midpoint', 'owners', 'log_ownership',
    'is_multiplayer', 'has_controller_support',
    'player_engagement', 'avg_recent_players',
    'days_since_release', 'release_year'
]"""
new_gfeats = """game_feats = [
    'appid', 'title', 'developer',
    'primary_genre', 'price_tier',
    'review_score', 'total_reviews', 'log_total_reviews',
    'ownership_midpoint', 'log_ownership',
    'is_multiplayer', 'has_controller_support',
    'player_engagement', 'avg_recent_players',
    'days_since_release', 'release_year', 'is_legacy',
]"""
if old_gfeats not in src:
    raise SystemExit("73414044: game_feats anchor not found")
src = src.replace(old_gfeats, new_gfeats, 1)
set_source(cell, src)
print("cell 73414044: game_feats updated (-developer_tier, -owners; +log_total_reviews, +is_legacy)")

# ---- e218a807: cols_part2 cleanup ----
cell = get_cell(nb, "e218a807")
src = "".join(cell["source"])
old_cp2 = """cols_part2 = [
    'appid', 'title', 'developer',
    'start_date', 'end_date', 'sale_duration_days',
    'discount_depth', 'discount_percent', 'sale_type',
    'age_at_sale_days', 'days_since_last_sale', 'cumulative_sale_count',
    'days_since_release', 'release_year',
    'primary_genre', 'developer_tier', 'price_tier',
    'review_score', 'total_reviews',
    'ownership_midpoint', 'owners', 'log_ownership',
    'is_multiplayer', 'has_controller_support',
    'player_engagement', 'avg_recent_players',
    'player_count_baseline', 'baseline_players',
    'player_count_during_sale', 'during_players',
    'uplift_percent', 'uplift_tier', 'sale_effectiveness_tier'
]"""
new_cp2 = """cols_part2 = [
    'appid', 'title', 'developer',
    'start_date', 'end_date', 'sale_duration_days',
    'discount_depth', 'discount_percent', 'sale_type',
    'age_at_sale_days', 'days_since_last_sale', 'cumulative_sale_count',
    'days_since_release', 'release_year', 'is_legacy',
    'primary_genre', 'price_tier',
    'review_score', 'total_reviews', 'log_total_reviews',
    'ownership_midpoint', 'log_ownership',
    'is_multiplayer', 'has_controller_support',
    'player_engagement', 'avg_recent_players',
    'player_count_baseline', 'baseline_players',
    'player_count_during_sale', 'during_players',
    'uplift_percent', 'uplift_tier', 'sale_effectiveness_tier',
]"""
if old_cp2 not in src:
    raise SystemExit("e218a807: cols_part2 anchor not found")
src = src.replace(old_cp2, new_cp2, 1)
set_source(cell, src)
print("cell e218a807: cols_part2 cleaned")

# ---- 1cf27364: panel feature merge -- drop developer_tier ----
cell = get_cell(nb, "1cf27364")
src = "".join(cell["source"])
old_panel = """panel_features = panel.merge(
    df[[
        'appid', 'title',
        'initial_price',
        'primary_genre', 'developer_tier', 'price_tier',
        'review_score', 'log_ownership',
        'is_multiplayer', 'has_controller_support', 'achievements_total',
        'publisher',
    ]],
    on='appid',
    how='left',
)"""
new_panel = """panel_features = panel.merge(
    df[[
        'appid', 'title',
        'initial_price', 'log_initial_price',
        'primary_genre', 'price_tier',
        'review_score', 'log_ownership',
        'is_multiplayer', 'has_controller_support',
        'achievements_total', 'log_achievements_total',
        'publisher',
    ]],
    on='appid',
    how='left',
)"""
if old_panel not in src:
    raise SystemExit("1cf27364: panel merge anchor not found")
src = src.replace(old_panel, new_panel, 1)
set_source(cell, src)
print("cell 1cf27364: panel merge updated (-developer_tier; +log_initial_price, +log_achievements_total)")

NB02.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\n[notebook 02] saved.")

# =============================================================================
# Notebook 04 patch: cell 02ef2e3c -- remove dominant_dev pull
# =============================================================================
nb = json.loads(NB04.read_text(encoding="utf-8"))
cell = get_cell(nb, "02ef2e3c")
src = "".join(cell["source"])
old_dominant_dev = "    dominant_dev   = members['developer_tier'].mode().iloc[0]   if not members['developer_tier'].mode().empty   else 'Unknown'\n"
if old_dominant_dev not in src:
    print("WARN: 02ef2e3c: dominant_dev line not found verbatim (may have been edited)")
else:
    src = src.replace(old_dominant_dev, "")
    set_source(cell, src)
    NB04.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[notebook 04] cell 02ef2e3c: dominant_dev pull removed.")

# =============================================================================
# Database: drop empty player_counts table
# =============================================================================
con = sqlite3.connect(DB)
n_rows = con.execute("SELECT COUNT(*) FROM player_counts").fetchone()[0]
if n_rows > 0:
    print(f"WARN: player_counts has {n_rows} rows -- skipping DROP for safety")
else:
    con.execute("DROP TABLE IF EXISTS player_counts")
    con.commit()
    print(f"[database] DROP TABLE player_counts (was empty).")
con.close()

# =============================================================================
# src/db.py: remove player_counts schema
# =============================================================================
db_py = SRC_DB.read_text(encoding="utf-8")
if "player_counts" in db_py:
    print(f"[src/db.py] player_counts is referenced -- check manually; this script does not auto-edit src/db.py")
    # Show the lines for the user
    for ln, line in enumerate(db_py.splitlines(), 1):
        if "player_counts" in line.lower():
            print(f"  L{ln}: {line.strip()}")
else:
    print(f"[src/db.py] no player_counts references found.")

print("\nAudit fixes applied. Notebook 03 will need separate updates -- it references")
print("developer_tier and value_retention_tier in EDA cells, which are now gone.")

"""Insert a fresh setup_backfills cell after setup_bootstrap in nb 04.

The original setup_backfills cell (added during the long-cell-split refactor)
was apparently removed. Without it, the new features in the NUM_FEATURES /
CAT_FEATURES lists (is_legacy, log_average_forever, log_average_2weeks,
log_days_until_first_sale) would not exist on `df`.

This script idempotently inserts a comprehensive backfill cell that covers:
  - All the original audit-era backfills (ever_discounted, achievements_total,
    max_discount_ever, publisher) in case cleaned_games predates the audit.
  - The four new features for Step 4.5.
"""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

BACKFILL_SRC = '''# Backfills -- ensure df has every column the modeling cells need, even if
# `cleaned_games` is from an older snapshot. After re-running notebook 02
# end-to-end, most of these become no-ops thanks to the `if not in columns`
# guards.

# ---- Audit-era backfills (also in case 02 hasn't been re-run) ------------
if 'ever_discounted' not in df.columns:
    ever_disc = pd.read_sql_query(
        'SELECT appid, MAX(CASE WHEN cut > 0 THEN 1 ELSE 0 END) AS ever_discounted '
        'FROM price_history GROUP BY appid', conn,
    )
    df = df.merge(ever_disc, on='appid', how='left')

if 'achievements_total' not in df.columns:
    ach = pd.read_sql_query('SELECT appid, achievements_total FROM games', conn)
    df = df.merge(ach, on='appid', how='left')

if 'max_discount_ever' not in df.columns:
    mde = pd.read_sql_query(
        'SELECT appid, MAX(cut) AS max_discount_ever '
        'FROM price_history GROUP BY appid', conn,
    )
    df = df.merge(mde, on='appid', how='left')

if 'publisher' not in df.columns:
    pub = pd.read_sql_query('SELECT appid, publisher FROM games', conn)
    df = df.merge(pub, on='appid', how='left')

df['ever_discounted']    = df['ever_discounted'].fillna(0).astype(int)
df['achievements_total'] = df['achievements_total'].fillna(0).astype(int)
df['max_discount_ever']  = df['max_discount_ever'].fillna(0).astype(float)
df['publisher']          = df['publisher'].fillna('Unknown').astype(str)

# ---- Extra-signal features for Step 4.5 ordinal classifier ---------------
if 'is_legacy' not in df.columns:
    df['is_legacy'] = (df['days_since_release'] > 1825).astype(int)

if 'log_average_forever' not in df.columns:
    spy = pd.read_sql_query(
        'SELECT appid, average_forever, average_2weeks FROM steamspy', conn,
    )
    df = df.merge(spy, on='appid', how='left')
    df['average_forever']      = df['average_forever'].fillna(0).clip(lower=0)
    df['average_2weeks']       = df['average_2weeks'].fillna(0).clip(lower=0)
    df['log_average_forever']  = np.log1p(df['average_forever'])
    df['log_average_2weeks']   = np.log1p(df['average_2weeks'])

if 'log_days_until_first_sale' not in df.columns:
    first_sale = pd.read_sql_query("""
        SELECT appid, MIN(timestamp) AS first_sale_timestamp
        FROM price_history WHERE cut > 0 GROUP BY appid
    """, conn)
    first_sale['first_sale_dt'] = pd.to_datetime(first_sale['first_sale_timestamp'], utc=True)
    df = df.merge(first_sale[['appid', 'first_sale_dt']], on='appid', how='left')
    df['_release_dt']           = pd.to_datetime(df['release_date'], errors='coerce', utc=True)
    days_until                  = (df['first_sale_dt'] - df['_release_dt']).dt.days
    df['days_until_first_sale']     = days_until.fillna(-1).astype(int)
    df['log_days_until_first_sale'] = np.log1p(df['days_until_first_sale'].clip(lower=0))
    df = df.drop(columns=['_release_dt', 'first_sale_dt'])

print('Backfills complete.')
print(f"  ever_discounted:           {df['ever_discounted'].sum():,} of {len(df):,}")
print(f"  is_legacy:                 {df['is_legacy'].sum():,} games are >5y old")
print(f"  log_average_forever:       median {df['log_average_forever'].median():.2f}")
print(f"  log_average_2weeks:        median {df['log_average_2weeks'].median():.2f}")
print(f"  log_days_until_first_sale: median {df['log_days_until_first_sale'].median():.2f}")
'''

ast.parse(BACKFILL_SRC)

# Idempotent: if a setup_backfills cell already exists, refresh it; else insert
existing_idx = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == "setup_backfills"), None)
new_cell = {
    "cell_type":       "code",
    "id":              "setup_backfills",
    "metadata":        {},
    "source":          BACKFILL_SRC.splitlines(keepends=True),
    "outputs":         [],
    "execution_count": None,
}

if existing_idx is not None:
    nb["cells"][existing_idx] = new_cell
    print(f"setup_backfills already existed at idx {existing_idx}; source refreshed.")
else:
    bootstrap_idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "setup_bootstrap")
    nb["cells"].insert(bootstrap_idx + 1, new_cell)
    print(f"setup_backfills inserted at idx {bootstrap_idx + 1} (after setup_bootstrap).")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Notebook now has {len(nb['cells'])} cells.")

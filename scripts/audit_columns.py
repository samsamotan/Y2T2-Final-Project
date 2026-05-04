"""Audit notebook 04 column references against the cleaned tables.

For every string subscript like df['col'], df_model['col'], panel['col']:
  - If the column is in cleaned_games / cleaned_discount_panel, OK.
  - If it's later assigned in the notebook (df['col'] = ...), it's derived in-notebook.
  - Otherwise, flag as a missing column.

Run this whenever you change cleaned_games schema or 04's column references.
"""
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
DB_PATH = ROOT / "data" / "steam.db"

# ---- Tables 04 reads from ----
conn = sqlite3.connect(DB_PATH)
schema = {}
for tbl in ("cleaned_games", "cleaned_discount_panel", "cleaned_sale_events"):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    schema[tbl] = set(cols)
    print(f"  {tbl:<30}  {len(cols)} columns")
conn.close()

# All columns 04 could legitimately pull from any cleaned table (current DB state):
all_cleaned_cols = schema["cleaned_games"] | schema["cleaned_discount_panel"] | schema["cleaned_sale_events"]

# ever_discounted is derived in 04's setup cell from raw price_history
derived_at_load = {"ever_discounted"}

# ---- What WILL exist after re-running 02 ----
# Parse 02's code to find:
#   - cols_part1 (what gets saved into cleaned_games)
#   - cols_part2 (what gets saved into cleaned_sale_events)
#   - panel table columns (what gets saved into cleaned_discount_panel)
nb02 = json.loads((ROOT / "notebooks" / "02_data_cleaning.ipynb").read_text(encoding="utf-8"))
nb02_src = "\n".join("".join(c["source"]) for c in nb02["cells"] if c["cell_type"] == "code")

def extract_list(var_name: str, src: str) -> set[str]:
    m = re.search(rf"{var_name}\s*=\s*\[([^\]]+)\]", src)
    if not m:
        return set()
    return set(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))

future_cleaned_games  = extract_list("cols_part1", nb02_src)
future_sale_events    = extract_list("cols_part2", nb02_src)
# panel_features has every column from `panel` + game attrs — collect via assignment scan
future_panel_cols = set()
# The .agg() call uses nested parens — `name = ('col', 'func')` — which a
# bracket-balanced regex would mishandle. Just scan the cell for `name = (`
# patterns inside an .agg(...) and trust that's the agg's keyword args.
agg_block = re.search(r"\.groupby\(\[['\"]appid['\"],\s*['\"]age_year['\"]\]\)\.agg\((.*?)\)\.reset_index\(\)", nb02_src, re.DOTALL)
if agg_block:
    future_panel_cols.update(re.findall(r"(\w+)\s*=\s*\(", agg_block.group(1)))
future_panel_cols.update({"appid", "age_year"})  # always present
# Also pick up the merged game attributes from the panel-feature cell
panel_merge = re.search(r"panel_features\s*=\s*panel\.merge\([^)]*?df\[\[([^\]]+)\]\]", nb02_src, re.DOTALL)
if panel_merge:
    future_panel_cols.update(re.findall(r"['\"]([^'\"]+)['\"]", panel_merge.group(1)))
future_panel_cols.add("buyer_value_at_age")  # explicitly assigned

future_all = future_cleaned_games | future_sale_events | future_panel_cols

# ---- Parse 04 ----
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

DF_NAMES = r"(?:df|df_model|df_model_clean|model_df|raw|panel|panel_clean|panel_features|X|X_train_reg|X_test_reg|X_train_cls|X_test_cls)"
SUBSCRIPT = re.compile(rf"{DF_NAMES}\s*\[\s*['\"]([^'\"]+)['\"]\s*\]")
ASSIGNMENT = re.compile(rf"{DF_NAMES}\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=")
DROPNA_SUBSET = re.compile(r"dropna\([^)]*subset\s*=\s*\[([^\]]+)\]")
LIST_LITERAL_STR = re.compile(r"['\"]([^'\"]+)['\"]")

referenced: dict[str, list[int]] = {}
assigned_in_nb: set[str] = set()
list_features: set[str] = set()  # columns referenced via num_features = [..] etc.

for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])

    # 1. Direct subscripts
    for col in SUBSCRIPT.findall(src):
        referenced.setdefault(col, []).append(i)
    # 2. Assignments — column is being created/modified in the notebook
    for col in ASSIGNMENT.findall(src):
        assigned_in_nb.add(col)
    # 3. dropna(subset=[...]) — strings inside the list are columns referenced
    for match in DROPNA_SUBSET.findall(src):
        for col in LIST_LITERAL_STR.findall(match):
            referenced.setdefault(col, []).append(i)
    # 4. num_features / cat_features lists
    for var in ("num_features", "cat_features", "panel_num_features", "panel_cat_features",
                "STRATEGY_FEATURES", "FEATURES", "FEAT_LABELS", "BASE_FEATURES"):
        m = re.search(rf"{var}\s*=\s*\[([^\]]+)\]", src)
        if m:
            for col in LIST_LITERAL_STR.findall(m.group(1)):
                list_features.add(col)
                referenced.setdefault(col, []).append(i)

# ---- Categorize ----
def col_status(col):
    """Return one of: 'present_now', 'present_after_02_rerun', 'derived', 'missing'."""
    if col in all_cleaned_cols:        return "present_now"
    if col in assigned_in_nb:          return "derived"
    if col in derived_at_load:         return "derived"
    if col in future_all:              return "present_after_02_rerun"
    return "missing"

bucketed = {"present_now": [], "present_after_02_rerun": [], "derived": [], "missing": []}
for col, cells in referenced.items():
    bucketed[col_status(col)].append((col, cells))
missing = bucketed["missing"]
fixable_by_rerun = bucketed["present_after_02_rerun"]

print()
print("=" * 70)
print(f"Notebook 04 column audit")
print("=" * 70)
print(f"  Total distinct columns referenced     : {len(referenced):>4}")
print(f"  Present in current DB                 : {len(bucketed['present_now']):>4}")
print(f"  Derived in notebook (assignments)     : {len(bucketed['derived']):>4}")
print(f"  Will exist AFTER re-running 02        : {len(fixable_by_rerun):>4}")
print(f"  [MISSING] No source found anywhere    : {len(missing):>4}")

if fixable_by_rerun:
    print()
    print("[FIXABLE] These columns are missing from the current DB but ARE produced")
    print("by 02's current code. Re-run 02 to populate them:")
    for col, cells in sorted(fixable_by_rerun):
        target = ("cleaned_discount_panel" if col in future_panel_cols
                  else "cleaned_games"      if col in future_cleaned_games
                  else "cleaned_sale_events")
        print(f"  - {col!r}  -> {target}  (referenced in cells {cells})")

if missing:
    print()
    print("[MISSING] Columns referenced by 04 but NOT produced by any source:")
    for col, cells in sorted(missing):
        print(f"  - {col!r}  (referenced in cells {cells})")
else:
    print()
    print("[OK] No real gaps. Every referenced column is either in a cleaned")
    print("table, derived in the notebook, or will be after re-running 02.")

# Bonus: list cleaned_games columns that 04 doesn't use — informational
unused = schema["cleaned_games"] - referenced.keys()
if unused:
    print()
    print(f"Note: cleaned_games has {len(unused)} columns 04 doesn't reference. Not a problem,")
    print("just a flag if you want to slim cleaned_games or use them as new features:")
    for col in sorted(unused):
        print(f"  - {col}")

"""Drop `price_tier` from notebook 04's modeling feature lists.

Two surgical edits:
  - cell `ccb410d0` (game-level preprocessor, Steps 4.4 / 4.5):
        remove 'price_tier' from CAT_FEATURES
  - cell `d9c7a96a` (panel preprocessor, Step 4.6):
        remove 'price_tier' from panel_cat_features

Leaves cell `02ef2e3c` (archetype labelling) alone -- it uses price_tier as a
narrative/modal-attr label, not as a model feature, and cleaned_games still
carries the column from notebook 02. The archetype-inference grid will keep
'price_tier' as a record field but it'll be ignored during predict() because
the panel preprocessor no longer asks for it.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"

EDITS = [
    ("ccb410d0", "CAT_FEATURES", "    'price_tier',\n"),
    ("d9c7a96a", "panel_cat_features", "    'price_tier',\n"),
]

nb = json.loads(NB.read_text(encoding="utf-8"))
changed = []
for cell_id, varname, line_to_drop in EDITS:
    cell = next((c for c in nb["cells"] if c.get("id") == cell_id), None)
    if cell is None:
        raise SystemExit(f"Cell {cell_id} not found")
    src = cell["source"]
    if line_to_drop not in src:
        print(f"!! cell {cell_id}: line not found verbatim, skipping")
        continue
    # Sanity: confirm it's inside the variable's list, not somewhere else
    pre = "".join(src[: src.index(line_to_drop)])
    if varname not in pre:
        raise SystemExit(f"cell {cell_id}: removed line not under {varname}")
    src.remove(line_to_drop)
    cell["source"] = src
    changed.append(cell_id)

# Compile-check: stitch together the two edited cells and exec-parse them.
# We can't actually run them (would need pandas + the DB), but a parse pass
# catches any obvious syntax damage.
import ast
for cell_id in changed:
    cell = next(c for c in nb["cells"] if c.get("id") == cell_id)
    src_str = "".join(cell["source"])
    ast.parse(src_str)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Edited cells: {changed}. Syntax OK.")

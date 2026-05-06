"""Confirm nb 04 is back on 5% bins and has the new features wired in."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

# Check step45_bins is back on 5%
bins = next(c for c in nb["cells"] if c.get("id") == "step45_bins")
bins_src = "".join(bins["source"])
assert "np.arange(0, 1.05, 0.05)" in bins_src, "step45_bins NOT reverted to 5%"
assert "STEAM" not in bins_src.upper().replace("STREAM", ""), "Steam-tier text still present"
print("step45_bins: 5% bin formula present, Steam-tier text removed -- OK")

# Check setup_backfills has the new features
sb = next(c for c in nb["cells"] if c.get("id") == "setup_backfills")
sb_src = "".join(sb["source"])
for needle in ("is_legacy", "log_average_forever", "log_average_2weeks", "log_days_until_first_sale"):
    assert needle in sb_src, f"backfill missing: {needle}"
print("setup_backfills: all four new features have backfills -- OK")

# Check prep_features has the new features in the lists
pf = next(c for c in nb["cells"] if c.get("id") == "prep_features")
pf_src = "".join(pf["source"])
import re
m = re.search(r"NUM_FEATURES\s*=\s*\[([^\]]+)\]", pf_src, re.DOTALL)
num_block = m.group(1)
for needle in ("log_average_forever", "log_average_2weeks", "log_days_until_first_sale"):
    assert needle in num_block, f"NUM_FEATURES missing: {needle}"
m = re.search(r"CAT_FEATURES\s*=\s*\[([^\]]+)\]", pf_src, re.DOTALL)
cat_block = m.group(1)
assert "is_legacy" in cat_block, "CAT_FEATURES missing is_legacy"
print("prep_features: NUM/CAT lists updated -- OK")

# Compile-check every code cell (excluding Jupyter magics)
err_count = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if any(ln.lstrip().startswith(("!", "%")) for ln in src.splitlines()):
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  PARSE ERROR cell[{i}] id={c.get('id')}: {e}")
        err_count += 1
print(f"\nAll code cells parse cleanly: {err_count} errors")
print(f"\nNotebook 04: {len(nb['cells'])} cells")

import ast
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "05_ml_part2_sale_effectiveness.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src_full = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

# Pull just the BASE_FEATURES list block
m = re.search(r"BASE_FEATURES\s*=\s*\[(.*?)\]", src_full, re.DOTALL)
features_block = m.group(1) if m else ""

print("Sanity checks:")
checks = [
    ("log_total_reviews exists",              "log_total_reviews" in src_full),
    ("log_achievements_total exists",         "log_achievements_total" in src_full),
    ("BASE_FEATURES has log_achievements",    "log_achievements_total" in features_block),
    ("BASE_FEATURES has log_total_reviews",   "log_total_reviews" in features_block),
    ("BASE_FEATURES drops raw achievements",  "'achievements_total'" not in features_block),
]
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL':<4}  {name}")

for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("\nAll code cells parse cleanly.")

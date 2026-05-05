import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04c_ml_part1_smote.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
print(f"{NB.name} has {len(nb['cells'])} cells")

train = next(c for c in nb["cells"] if c.get("id") == "step45_train")
src = "".join(train["source"])
ast.parse(src)
assert "from imblearn.over_sampling import SMOTE" in src
assert "ImbPipeline" in src
assert "k_neighbors=3" in src
assert "pred_ord" in src
print("  step45_train: SMOTE imports + ImbPipeline + k=3 + pred_ord -- OK")

title = "".join(nb["cells"][0]["source"])
assert "SMOTE" in title
print("  title:        SMOTE variant note -- OK")

# Confirm the downstream plot cells still reference the same variable names
for cid in ("step45_plot", "step45_total_norm"):
    cell = next(c for c in nb["cells"] if c.get("id") == cid)
    cs = "".join(cell["source"])
    assert "pred_ord" in cs and "y_test_ord" in cs, f"{cid} variable mismatch"
print("  downstream plot cells: still use pred_ord / y_test_ord -- OK")

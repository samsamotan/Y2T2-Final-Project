import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

nb = json.loads((ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb").read_text(encoding="utf-8"))
cell_ids = [c.get("id") for c in nb["cells"]]
assert "save_model_artifact" in cell_ids
i = cell_ids.index("save_model_artifact")
print(f"nb 04: save_model_artifact at idx {i}, follows {cell_ids[i - 1]!r}")

nb = json.loads((ROOT / "notebooks" / "07_predict_single_game.ipynb").read_text(encoding="utf-8"))
print(f"\nnb 07: {len(nb['cells'])} cells")
for c in nb["cells"]:
    pv = "".join(c["source"]).splitlines()[0][:75].encode("ascii", "replace").decode("ascii")
    print(f"  {c['cell_type']:<8}  {c.get('id','?'):<14}  {pv}")

for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("\nAll code cells parse cleanly.")

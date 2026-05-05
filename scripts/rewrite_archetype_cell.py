"""Rewrite cell `02ef2e3c` in notebook 04 with the trajectory-clustering version
(Option A from the design discussion).

The new cell source lives in `_archetype_cell_source.py` -- editing that file
is much less error-prone than embedding a triple-quoted Python source inside
another Python script.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB        = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
SOURCE    = ROOT / "scripts" / "_archetype_cell_source.py"
CELL_ID   = "02ef2e3c"

src = SOURCE.read_text(encoding="utf-8")
ast.parse(src)   # compile-check before touching the notebook

nb = json.loads(NB.read_text(encoding="utf-8"))
cell = next((c for c in nb["cells"] if c.get("id") == CELL_ID), None)
if cell is None:
    raise SystemExit(f"Cell {CELL_ID} not found")

cell["source"]          = src.splitlines(keepends=True)
cell["outputs"]         = []
cell["execution_count"] = None

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Rewrote cell {CELL_ID}: {len(src.splitlines())} lines, {len(src)} chars. Syntax OK.")

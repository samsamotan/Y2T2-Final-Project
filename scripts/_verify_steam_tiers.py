"""Verify Steam-tier bins are wired in for both nb 04 and nb 04c."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for fn in ("04_ml_part1_price_and_value_retention.ipynb",
           "04c_ml_part1_smote.ipynb"):
    nb_path = ROOT / "notebooks" / fn
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    print(f"--- {fn} ---")
    for cid in ("step45_bins", "step45_plot", "step45_total_norm"):
        cell = next((c for c in nb["cells"] if c.get("id") == cid), None)
        if cell is None:
            print(f"  cell {cid}: MISSING")
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
            ok = True
        except SyntaxError as e:
            ok = False
            print(f"  cell {cid}: PARSE ERROR -- {e}")
        has_steam = "Steam-aligned" in src or "Steam-native" in src or "Steam-tier" in src or "STEAM" in src.upper()
        has_old   = "np.arange(0, 1.05, 0.05)" in src
        n_bins_in_src = "N_BINS      = 13" in src or "N_BINS      = len(BIN_EDGES) - 1" in src
        print(f"  cell {cid}: parse={ok}, mentions Steam tiers={has_steam}, "
              f"old 5% bin still in src={has_old}, irregular N_BINS={n_bins_in_src}")
    print()

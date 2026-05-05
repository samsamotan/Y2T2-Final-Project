"""Delete notebook 04b and update the two stale cross-references in 04 cell 14.

  - References "[04b_ml_part1_ordinal_classification.ipynb](...)" -> "Step 4.5
    of this notebook"
  - "see also 04b for the explicit binned framing" -> "see Step 4.5 below"
  - "...continuous regression in 04 and an ordinal classifier in 04b..." ->
    "...continuous regression in Step 4.4 and an ordinal classifier in Step 4.5..."

Then deletes notebooks/04b_ml_part1_ordinal_classification.ipynb.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB04 = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
NB04B = ROOT / "notebooks" / "04b_ml_part1_ordinal_classification.ipynb"

nb = json.loads(NB04.read_text(encoding="utf-8"))
cell = next(c for c in nb["cells"] if c.get("id") == "219e6183")
src = "".join(cell["source"])

# Pin each replacement to a unique substring to avoid collateral edits.
replacements = [
    (
        "**An ordinal classification reframe lives in [04b_ml_part1_ordinal_classification.ipynb](04b_ml_part1_ordinal_classification.ipynb)** - same target binned at 5% intervals, evaluated with bin-distance metrics that match how buyers actually use predictions. Run that notebook for the tier-aware view; this notebook stays focused on the continuous regression.",
        "**An ordinal classification reframe lives in Step 4.5 below** - same target binned at 5% intervals, evaluated with bin-distance metrics that match how buyers actually use predictions. Step 4.4 stays focused on the continuous regression; Step 4.5 reframes the same target as 20 ordinal tiers.",
    ),
    (
        "see also 04b for the explicit binned framing",
        "see Step 4.5 for the explicit binned framing",
    ),
    (
        '"We predict the deepest discount each game has reached, with a continuous regression in 04 and an ordinal classifier in 04b.',
        '"We predict the deepest discount each game has reached, with a continuous regression in Step 4.4 and an ordinal classifier in Step 4.5.',
    ),
]

missing = [old[:60] for old, _ in replacements if old not in src]
if missing:
    raise SystemExit(f"Could not find the following snippets verbatim:\n  - " + "\n  - ".join(missing))

for old, new in replacements:
    src = src.replace(old, new, 1)

# Sanity: confirm 04b is no longer referenced in the cell.
if "04b" in src:
    raise SystemExit("'04b' still appears in cell 219e6183 after replacements -- aborting.")

cell["source"] = src.splitlines(keepends=True)
NB04.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("notebook 04 cell 219e6183: 04b references rewritten to Step 4.4 / Step 4.5.")

# Now delete the 04b notebook
if NB04B.exists():
    NB04B.unlink()
    print(f"deleted: {NB04B.name}")
else:
    print(f"already gone: {NB04B.name}")

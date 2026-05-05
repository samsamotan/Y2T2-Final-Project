"""Build a SMOTE variant of notebook 04.

Source: notebooks/04_ml_part1_price_and_value_retention.ipynb
Output: notebooks/04c_ml_part1_smote.ipynb

What changes:
  - Title markdown gets a "(SMOTE variant)" suffix and a short note
  - Step 4.5 train cell (`step45_train`) is replaced with a SMOTE-augmented
    version that uses imblearn's pipeline so SMOTE only fires during .fit()
  - Step 4.5 discussion cell gets a few SMOTE-specific lines

Variable names in the SMOTE training cell match the baseline (X_train_ord,
X_test_ord, y_train_ord, y_test_ord, pred_ord, ordinal_results) so downstream
cells (step45_plot, step45_total_norm) work unchanged.

SMOTE k_neighbors=3 -> requires >=4 samples per class. Bins with fewer samples
are filtered out before training and the drop is reported.
"""
import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
DST  = ROOT / "notebooks" / "04c_ml_part1_smote.ipynb"

shutil.copy2(SRC, DST)
nb = json.loads(DST.read_text(encoding="utf-8"))

# ---- 1. Title markdown ---------------------------------------------------
title_cell = nb["cells"][0]
assert title_cell["cell_type"] == "markdown"
title_cell["source"] = [
    "# 04c - ML Model Development: Part 1 (SMOTE variant)\n",
    "\n",
    "Same pipeline as `04_ml_part1_price_and_value_retention.ipynb` with one change: **Step 4.5 (Ordinal Classification) applies SMOTE oversampling** to the imbalanced 20-bin discount target before training the Random Forest classifier.\n",
    "\n",
    "Everything else (Step 4.4 regression, Step 4.6 panel, Step 4.7 buy-time recommender) is identical to the baseline notebook.\n",
    "\n",
    "Compare the Step 4.5 confusion matrix and bin-distance metrics here against the baseline notebook to see SMOTE's effect on minority-bin recall.\n",
]

# ---- 2. step45_train: SMOTE-augmented training -------------------------
new_train = '''# Step 4.5 (SMOTE variant): apply SMOTE to balance the 20-bin target before
# training. imblearn's Pipeline knows to skip the resampling step at predict
# time, so the test set keeps the true class distribution.
#
# SMOTE k_neighbors=3 -> needs >=4 training samples per class. We filter out
# very rare bins (e.g., 0-5%, 95-100%) where synthesis is too sparse to be
# meaningful and SMOTE would error otherwise.

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

X_ord = df_clean[NUM_FEATURES + CAT_FEATURES + PUB_FEATURE]
y_ord = df_clean['discount_bin']

# Drop bins that are too sparse for SMOTE k_neighbors=3.
bin_counts   = y_ord.value_counts()
keep_bins    = bin_counts[bin_counts >= 4].index.tolist()
dropped_bins = bin_counts[bin_counts < 4]

if len(dropped_bins) > 0:
    print(f'Dropping {len(dropped_bins)} bins with <4 samples (SMOTE cannot synthesize):')
    for b, n in dropped_bins.items():
        lo, hi = int(BIN_EDGES[b] * 100), int(BIN_EDGES[b + 1] * 100)
        print(f'  bin {b:>2} ({lo:>3}-{hi:>3}%): {n} sample(s)')

mask  = y_ord.isin(keep_bins)
X_ord = X_ord[mask]
y_ord = y_ord[mask]
print(f'\\nTraining on {y_ord.nunique()} bins ({len(y_ord):,} samples).')

# Stratified split BEFORE resampling -- test set should reflect true imbalance.
X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(
    X_ord, y_ord, test_size=0.2, random_state=42, stratify=y_ord,
)

# imblearn pipeline: prep -> SMOTE -> RF. Note we drop class_weight here because
# SMOTE rebalances the training data; double-weighting via the model would be
# redundant.
rf_ord = ImbPipeline([
    ('prep',  preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )),
])
rf_ord.fit(X_train_ord, y_train_ord)
pred_ord = rf_ord.predict(X_test_ord)

# Report SMOTE's effective training distribution
prep_step  = rf_ord.named_steps['prep']
smote_step = rf_ord.named_steps['smote']
X_train_pre   = prep_step.transform(X_train_ord)
X_train_resampled, y_train_resampled = smote_step.fit_resample(X_train_pre, y_train_ord)
print(f'After SMOTE: {len(y_train_resampled):,} training samples '
      f'({len(y_train_resampled) - len(y_train_ord):,} synthetic), '
      f'{y_train_resampled.value_counts().min():,} per class minimum.')

# Bin-distance metrics -- ordinal-aware.
distances = np.abs(pred_ord - y_test_ord.values)
ordinal_results = pd.DataFrame([{
    'Exact (within 0%)':           (distances == 0).mean(),
    'Within +/- 1 bin (5%)':       (distances <= 1).mean(),
    'Within +/- 2 bins (10%)':     (distances <= 2).mean(),
    'Within +/- 4 bins (20%)':     (distances <= 4).mean(),
    'MABE (bins)':                 distances.mean(),
    'MABE (pp)':                   distances.mean() * 5,
}], index=['Random Forest (SMOTE)']).T

print('\\n=== Ordinal classification with SMOTE (Random Forest, 20 x 5% bins) ===')
display(ordinal_results.round(3))
'''
ast.parse(new_train)
train_cell = next(c for c in nb["cells"] if c.get("id") == "step45_train")
train_cell["source"]          = new_train.splitlines(keepends=True)
train_cell["outputs"]         = []
train_cell["execution_count"] = None

# ---- 3. step45_disc: SMOTE-specific discussion -------------------------
disc_cell = next(c for c in nb["cells"] if c.get("id") == "step45_disc")
disc_cell["source"] = [
    "### Ordinal classification with SMOTE - narrative\n",
    "\n",
    "**What changed vs the baseline notebook:**\n",
    "\n",
    "- Training data resampled with **SMOTE** (`k_neighbors=3`) so every retained bin contributes equal training mass.\n",
    "- Bins with **<4 samples** dropped before fitting (SMOTE can't synthesize on 1-3 points).\n",
    "- `class_weight` removed from the RF -- SMOTE replaces the weighting strategy.\n",
    "\n",
    "**What to compare against the baseline:**\n",
    "\n",
    "- **Within +/- 1 bin** -- this is the headline buyer-facing metric. Did SMOTE help or hurt?\n",
    "- **Off-diagonal mass in the confusion matrix** -- SMOTE often pulls minority-bin recall up at the cost of more false positives in adjacent bins, so the off-diagonal band may broaden.\n",
    "- **Diagonal share (total-normalized chart)** -- if SMOTE helped, the diagonal absolute share rises; if it overfit, the diagonal narrows because synthesized samples blur class boundaries.\n",
    "\n",
    "**SMOTE-specific caveats for the talk:**\n",
    "\n",
    "- SMOTE synthesizes by linear interpolation between minority-class neighbors. For an ordinal target this can produce samples that span unrealistic gaps -- a 20% off + 80% off interpolation is a 50% off synthesized point that may not correspond to any real game's behaviour.\n",
    "- The bin-distance metric is partly insulated from this (off-by-1 is still 'close'), but exact-match accuracy can degrade.\n",
    "- The dropped bins (<4 samples) are an inherent limit of synthesis-based oversampling on sparse ordinal data. You're trading the ability to predict tail bins for cleaner mid-bin recall.\n",
]

# ---- 4. Write the new notebook ----
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote: {DST.relative_to(ROOT)}  ({len(nb['cells'])} cells)")
print(f"  Title cell:        SMOTE variant note added")
print(f"  step45_train:      SMOTE pipeline (imblearn)")
print(f"  step45_disc:       SMOTE-specific discussion")

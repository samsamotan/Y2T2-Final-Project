"""Bring 04b's ordinal classification into 04 as Step 4.5.

Inserts 5 cells right before Step 4.6 (markdown cell '5947f771'):
  - Markdown: '## Step 4.5: Ordinal Classification (5% Discount Bins)'
  - Code: build the 20-bin target
  - Code: stratified split + RF + bin-distance metrics
  - Code: confusion matrix + error histogram (with new boundary-labeled axes)
  - Markdown: Discussion / caveats

Tick labels on the confusion matrix now show actual bin BOUNDARIES (0%, 10%,
20%, ..., 100%) at every other tick instead of bin centers.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB   = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
PANEL_HEADER_ID = "5947f771"   # Step 4.6 marker -- insert immediately before this

nb = json.loads(NB.read_text(encoding="utf-8"))

# ---------- 2. Build bins (code) ----------
bins_src = """\
# Build the 20-bin discount target. Each bin spans 5%.
# Reuses df_clean, NUM_FEATURES, CAT_FEATURES, PUB_FEATURE, preprocessor from Step 4.2.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BIN_EDGES   = np.arange(0, 1.05, 0.05)        # 0.00, 0.05, ..., 1.00
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_BINS      = len(BIN_EDGES) - 1

df_clean['discount_bin'] = pd.cut(
    df_clean['max_discount_ever'] / 100,
    bins=BIN_EDGES, labels=False, include_lowest=True,
).astype(int)

counts = df_clean['discount_bin'].value_counts().sort_index()
print(f'Target: {N_BINS} bins of 5% width each')
print(f'Populated bins: {len(counts)} / {N_BINS}')
print()
print('Bin distribution:')
for bin_id, n in counts.items():
    pct_low  = BIN_EDGES[bin_id]   * 100
    pct_high = BIN_EDGES[bin_id+1] * 100
    bar = '#' * max(1, int(n / counts.max() * 40))
    print(f'  bin {bin_id:>2} ({pct_low:>3.0f}-{pct_high:>3.0f}%):  {n:>4,}  {bar}')
"""

# ---------- 3. Train + metrics (code) ----------
train_src = """\
# Stratified split on the bin target, then train RF.
X_ord = df_clean[NUM_FEATURES + CAT_FEATURES + PUB_FEATURE]
y_ord = df_clean['discount_bin']

X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(
    X_ord, y_ord, test_size=0.2, random_state=42, stratify=y_ord,
)

rf_ord = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1,
    )),
])
rf_ord.fit(X_train_ord, y_train_ord)
pred_ord = rf_ord.predict(X_test_ord)

# Bin-distance metrics -- ordinal-aware (off-by-1 is much less bad than off-by-10).
distances = np.abs(pred_ord - y_test_ord.values)
ordinal_results = pd.DataFrame([{
    'Exact (within 0%)':           (distances == 0).mean(),
    'Within +/- 1 bin (5%)':       (distances <= 1).mean(),
    'Within +/- 2 bins (10%)':     (distances <= 2).mean(),
    'Within +/- 4 bins (20%)':     (distances <= 4).mean(),
    'MABE (bins)':                 distances.mean(),
    'MABE (pp)':                   distances.mean() * 5,
}], index=['Random Forest']).T

print('=== Ordinal classification (Random Forest, 20 x 5% bins) ===')
display(ordinal_results.round(3))
"""

# ---------- 4. Confusion matrix + error histogram, boundary-labeled axes ----------
plot_src = """\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion matrix (row-normalized) ---
cm = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                      normalize='true')   # each row sums to 1.0
im_cm = axes[0].imshow(cm, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
cbar = plt.colorbar(im_cm, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Proportion within actual bin', fontsize=9)

# Tick at every bin BOUNDARY (positions -0.5, 1.5, 3.5, ...) labeled with the
# actual % edge. Every other tick gets a label so the axis isn't crowded.
# imshow with origin='lower' centers bin i at integer y=i, so its lower edge
# sits at y=i-0.5 (i.e., at the boundary between bin i-1 and bin i).
boundary_positions = np.arange(-0.5, N_BINS, 2)
boundary_labels    = [f'{int(BIN_EDGES[i]*100)}%' for i in range(0, N_BINS + 1, 2)]
axes[0].set_xticks(boundary_positions); axes[0].set_xticklabels(boundary_labels)
axes[0].set_yticks(boundary_positions); axes[0].set_yticklabels(boundary_labels)
axes[0].set_xlabel('Predicted discount bin (5% wide)')
axes[0].set_ylabel('Actual discount bin (5% wide)')
axes[0].set_title('Confusion Matrix - Random Forest (20 bins, row-normalized)')
axes[0].plot([-0.5, N_BINS - 0.5], [-0.5, N_BINS - 0.5],
             'r--', linewidth=1, alpha=0.6, label='Perfect')
axes[0].legend(loc='upper left')

# --- Bin-distance error histogram ---
axes[1].hist(distances, bins=range(N_BINS + 1),
             color=PALETTE['cyan'], edgecolor='white')
axes[1].set_xlabel('Absolute bin distance (1 unit = 5%)')
axes[1].set_ylabel('Number of predictions')
axes[1].set_title('Prediction Error Distribution')
axes[1].axvline(distances.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Mean: {distances.mean():.2f} bins ({distances.mean()*5:.1f} pp off)')
axes[1].legend()

plt.tight_layout()
plt.show()
"""

# Compile-check all three code cells before we touch the notebook.
for src in (bins_src, train_src, plot_src):
    ast.parse(src)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "step45_hdr",
        "metadata": {},
        "source": [
            "> ## Step 4.5: Ordinal Classification (5% Discount Bins)\n",
            "\n",
            "We bin `max_discount_ever` into 20 ordinal classes of 5% each (0-5%, 5-10%, ..., 95-100%) and predict the bin directly. This is a discretized view of the same target Step 4.4 regresses continuously - the value comes from the **bin-distance metrics**, which translate cleanly into a buyer-facing claim:\n",
            "\n",
            "*\"The model predicts the peak discount within +/- 5 percentage points X% of the time.\"*\n",
        ],
    },
    {
        "cell_type": "code",
        "id": "step45_bins",
        "metadata": {},
        "source": bins_src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "id": "step45_train",
        "metadata": {},
        "source": train_src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "id": "step45_plot",
        "metadata": {},
        "source": plot_src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "step45_disc",
        "metadata": {},
        "source": [
            "### Ordinal classification - narrative\n",
            "\n",
            "The 20-bin RF maps each game to its predicted peak-discount tier. Buyer-facing reading:\n",
            "\n",
            "- **Within +/- 1 bin** is the headline number for the report - *\"the model lands within 5 percentage points of true X% of the time.\"*\n",
            "- **MABE (pp)** translates directly: *\"on average, the prediction is off by Y points.\"*\n",
            "\n",
            "**Caveats to mention in the talk:**\n",
            "\n",
            "- Bin distribution is heavily imbalanced - most games sit in the 50-90% range with very few in the extremes. `class_weight='balanced_subsample'` partially compensates, but tail bins (e.g., 0-5%, 95-100%) will still have weak per-bin accuracy.\n",
            "- Steam's actual discount values cluster at non-uniform tiers (25, 33, 50, 66, 75, 90 - not multiples of 5), so some 5% bins will always be near-empty by data structure rather than model error. That's why several rows of the confusion matrix appear pale - they have no test instances.\n",
            "- This classifier is a **discretized view** of the same target Step 4.4 regresses continuously. The two are complementary: the regression's R^2 is the headline predictive metric; the bin-distance metric is the buyer-facing translation.\n",
        ],
    },
]

# Insert immediately before the Step 4.6 marker
panel_idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == PANEL_HEADER_ID)
nb["cells"] = nb["cells"][:panel_idx] + new_cells + nb["cells"][panel_idx:]

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Inserted {len(new_cells)} cells before Step 4.6.")
print(f"Notebook now has {len(nb['cells'])} cells.")

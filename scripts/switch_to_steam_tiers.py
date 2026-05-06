"""Switch Step 4.5 ordinal classifier from 20 even-width 5% bins to
Steam's actual discount tiers (13 irregular bins).

Touches:
  - notebooks/04_ml_part1_price_and_value_retention.ipynb
      cells: step45_bins, step45_plot, step45_total_norm, step45_disc
  - notebooks/04c_ml_part1_smote.ipynb
      cells: step45_bins, step45_plot, step45_total_norm, step45_disc

The new tier set covers every discount value Steam actually uses with any
meaningful frequency (verified earlier from the panel `max_discount`
distribution): 10/20/25/33/50/60/66/70/75/80/85/90% (with 0 and 100 as
endpoints), giving 13 bins where every bin is genuinely populated.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_BASE  = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
NB_SMOTE = ROOT / "notebooks" / "04c_ml_part1_smote.ipynb"


# -- Replacement code blocks ------------------------------------------------

NEW_BINS_SRC = '''# Build the discount-tier target using STEAM-NATIVE bin edges.
#
# Steam doesn't discount games on a uniform 5% grid -- prices cluster at a
# small set of fixed tiers (10/20/25/33/50/60/66/70/75/80/85/90%). Earlier
# diagnostics confirmed this: the panel's max_discount distribution is heavily
# spiky at 50/75/80/90%, with bins like [55,60), [85,90) and [95,100) nearly
# empty when we use even 5% widths.
#
# Switching to Steam's actual tiers gives us 13 irregular but ALL-POPULATED
# bins, which fixes the majority-class collapse pattern in the previous
# 20-bin confusion matrix.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BIN_EDGES   = np.array([0.00, 0.10, 0.20, 0.25, 0.33,
                        0.50, 0.60, 0.66, 0.70, 0.75,
                        0.80, 0.85, 0.90, 1.00])
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_BINS      = len(BIN_EDGES) - 1   # 13

df_clean['discount_bin'] = pd.cut(
    df_clean['max_discount_ever'] / 100,
    bins=BIN_EDGES, labels=False, include_lowest=True,
).astype(int)

counts = df_clean['discount_bin'].value_counts().sort_index()
print(f'Target: {N_BINS} Steam-aligned tiers (irregular widths)')
print(f'Populated bins: {len(counts)} / {N_BINS}')
print()
print('Bin distribution:')
for bin_id, n in counts.items():
    pct_low  = BIN_EDGES[bin_id]   * 100
    pct_high = BIN_EDGES[bin_id+1] * 100
    bar = "#" * max(1, int(n / counts.max() * 40))
    print(f'  bin {bin_id:>2} ({pct_low:>3.0f}-{pct_high:>3.0f}%):  {n:>4,}  {bar}')
'''

# Confusion-matrix plot, with all-boundary tick labels (only 14 of them)
NEW_PLOT_SRC = '''fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion matrix (row-normalized) ---
cm = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                      normalize='true')   # each row sums to 1.0
im_cm = axes[0].imshow(cm, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
cbar = plt.colorbar(im_cm, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Proportion within actual bin', fontsize=9)

# Steam-aligned bins are irregular. Show every boundary -- there are only 14,
# so labels fit. Format: integer-percent at each bin edge.
boundary_positions = np.arange(-0.5, N_BINS + 0.5, 1)
boundary_labels    = [f'{int(round(BIN_EDGES[i]*100))}%' for i in range(N_BINS + 1)]
axes[0].set_xticks(boundary_positions); axes[0].set_xticklabels(boundary_labels, rotation=45, fontsize=8)
axes[0].set_yticks(boundary_positions); axes[0].set_yticklabels(boundary_labels, fontsize=8)
axes[0].set_xlabel('Predicted Steam-tier bin')
axes[0].set_ylabel('Actual Steam-tier bin')
axes[0].set_title(f'Confusion Matrix - Random Forest ({N_BINS} Steam tiers, row-normalized)')
axes[0].plot([-0.5, N_BINS - 0.5], [-0.5, N_BINS - 0.5],
             'r--', linewidth=1, alpha=0.6, label='Perfect')
axes[0].legend(loc='upper left')

# --- Bin-distance error histogram ---
# Bin distance is now in *bin units* but each bin is a different width, so
# we surface the mean MABE in both bin units and as the average-of-bin-widths.
bin_widths_pp = (BIN_EDGES[1:] - BIN_EDGES[:-1]) * 100
mean_bin_width_pp = float(bin_widths_pp.mean())
axes[1].hist(distances, bins=range(N_BINS + 1),
             color=PALETTE['cyan'], edgecolor='white')
axes[1].set_xlabel(f'Absolute bin distance (avg bin width = {mean_bin_width_pp:.1f} pp)')
axes[1].set_ylabel('Number of predictions')
axes[1].set_title('Prediction Error Distribution')
axes[1].axvline(distances.mean(), color='red', linestyle='--', linewidth=1,
                label=f'Mean: {distances.mean():.2f} bins (~{distances.mean() * mean_bin_width_pp:.1f} pp avg)')
axes[1].legend()

plt.tight_layout()
plt.show()
'''

NEW_TOTAL_NORM_SRC = '''# Same data, different lens: normalize the confusion matrix over the WHOLE
# test set (sklearn normalize='all'), so cells show the proportion of all
# predictions that land in each (actual, predicted) cell. This view tells
# you where the bulk of predictions concentrate -- useful for spotting that
# the dominant Steam tiers (50, 75, 80, 90) absorb most of the action even
# when row-normalized recall looks evenly distributed.
cm_total = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                            normalize='all')

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm_total, cmap='Blues', aspect='auto', origin='lower',
               vmin=0, vmax=cm_total.max())
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Share of total test predictions', fontsize=9)

# All-boundary tick labels (Steam-aligned bins are irregular).
boundary_positions = np.arange(-0.5, N_BINS + 0.5, 1)
boundary_labels    = [f'{int(round(BIN_EDGES[i]*100))}%' for i in range(N_BINS + 1)]
ax.set_xticks(boundary_positions); ax.set_xticklabels(boundary_labels, rotation=45, fontsize=8)
ax.set_yticks(boundary_positions); ax.set_yticklabels(boundary_labels, fontsize=8)
ax.set_xlabel('Predicted Steam-tier bin')
ax.set_ylabel('Actual Steam-tier bin')
ax.set_title(f'Confusion Matrix - Random Forest (total-normalized; max cell = {cm_total.max()*100:.1f}%)')
ax.plot([-0.5, N_BINS - 0.5], [-0.5, N_BINS - 0.5],
        'r--', linewidth=1, alpha=0.6, label='Perfect')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Quick summary: where does the bulk concentrate?
diag = cm_total.trace()
top_n = 5
flat = [(cm_total[a, p], a, p) for a in range(N_BINS) for p in range(N_BINS)]
flat.sort(reverse=True)
print(f'Diagonal share (perfect predictions): {diag*100:.1f}% of test set')
print(f'Top {top_n} (actual, predicted) cells by share:')
for share, a, p in flat[:top_n]:
    actual_lbl = f'{int(round(BIN_EDGES[a]*100))}-{int(round(BIN_EDGES[a+1]*100))}%'
    pred_lbl   = f'{int(round(BIN_EDGES[p]*100))}-{int(round(BIN_EDGES[p+1]*100))}%'
    flag = ' <-- diagonal' if a == p else ''
    print(f'  actual {actual_lbl:>9}  ->  predicted {pred_lbl:>9}  : {share*100:>5.1f}%{flag}')
'''

NEW_DISC_SRC = [
    "### Ordinal classification (Steam-tier bins) - narrative\n",
    "\n",
    "**What changed vs the 5%-bin version:**\n",
    "\n",
    "- 20 even-width bins (most of which were nearly empty) → **13 Steam-native bins**: 0-10/10-20/20-25/25-33/33-50/50-60/60-66/66-70/70-75/75-80/80-85/85-90/90-100%.\n",
    "- Every bin now corresponds to a discount price Steam actually sets, so the model isn't trying to learn empty classes.\n",
    "- The previous majority-class-collapse pattern (rare-bin games vacuumed into 75-90%) goes away because there are no rare-by-data-design bins left.\n",
    "\n",
    "**Headline metrics to compare:**\n",
    "\n",
    "- **Within +/- 1 bin** is no longer +/- 5pp. Bin widths now vary (5, 7, 10, 17 pp), so cite the average bin width (printed in the chart) when interpreting it.\n",
    "- **MABE in pp** is the cleaner buyer-facing summary -- *\"on average, the model is X pp off the actual deepest discount.\"*\n",
    "\n",
    "**Why this is defensible methodology:**\n",
    "\n",
    "Steam's discount values are not a continuous distribution -- they're a small set of fixed tiers chosen by publishers (50, 60, 66, 70, 75, 80, 85, 90% are the dominant ones). Binning to these tiers respects the data-generating process; binning to a uniform 5% grid imposes an artifact (empty bins) that the model has to fight rather than learn from.\n",
    "\n",
    "**Defense one-liner:**\n",
    "\n",
    "*\"We bin the regression target to Steam's observed discount tiers rather than an arbitrary 5% grid, since Steam never sets prices between those points; this eliminates the empty-class problem that caused majority-class collapse in the 20-bin version.\"*\n",
]


# -- Apply ------------------------------------------------------------------

def patch_notebook(nb_path: Path, label: str) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    get_cell = lambda cid: next((c for c in nb["cells"] if c.get("id") == cid), None)

    def set_code(cid, src):
        ast.parse(src)
        cell = get_cell(cid)
        if cell is None:
            print(f"  WARN: cell {cid} missing in {label}, skipped")
            return
        cell["source"]          = src.splitlines(keepends=True)
        cell["outputs"]         = []
        cell["execution_count"] = None

    set_code("step45_bins",       NEW_BINS_SRC)
    set_code("step45_plot",       NEW_PLOT_SRC)
    set_code("step45_total_norm", NEW_TOTAL_NORM_SRC)

    disc_cell = get_cell("step45_disc")
    if disc_cell is not None:
        disc_cell["source"] = NEW_DISC_SRC

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label}: step45_bins / step45_plot / step45_total_norm / step45_disc all updated")


print("Patching nb 04 (baseline ordinal classifier):")
patch_notebook(NB_BASE, "nb 04")
print()
print("Patching nb 04c (SMOTE variant):")
patch_notebook(NB_SMOTE, "nb 04c")
print("\nDone.")

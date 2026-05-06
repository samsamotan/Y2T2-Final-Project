"""Undo `revert_and_enrich_nb04.py`:
  - step45 cells: 5% bins -> Steam-tier bins (back to where Steam tiers lived)
  - prep_features: drop the four new features added to NUM/CAT lists
  - setup_backfills: never actually touched (cell was already missing), no-op

Restores nb 04 to its post-Steam-tier state, BEFORE the 5% revert + feature
enrich. Run optuna fresh after this to verify whether the 44pp result was a
caching/state issue rather than a model-quality issue.
"""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
get_cell = lambda cid: next((c for c in nb["cells"] if c.get("id") == cid), None)


def set_code(cid, src):
    cell = get_cell(cid)
    if cell is None:
        print(f"  WARN cell {cid} missing, skipped")
        return
    ast.parse(src)
    cell["source"]          = src.splitlines(keepends=True)
    cell["outputs"]         = []
    cell["execution_count"] = None


# ---- Steam-tier sources (the state we want to restore) -------------------

STEAM_BINS_SRC = '''# Build the discount-tier target using STEAM-NATIVE bin edges.
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

STEAM_PLOT_SRC = '''fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion matrix (row-normalized) ---
cm = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                      normalize='true')
im_cm = axes[0].imshow(cm, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
cbar = plt.colorbar(im_cm, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Proportion within actual bin', fontsize=9)

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

STEAM_TOTAL_NORM_SRC = '''cm_total = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                            normalize='all')

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm_total, cmap='Blues', aspect='auto', origin='lower',
               vmin=0, vmax=cm_total.max())
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Share of total test predictions', fontsize=9)

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

STEAM_DISC_MD = [
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


# Original NUM/CAT (without the four new features)
ORIG_NUM = """NUM_FEATURES = [
    'days_since_release',
    'log_initial_price',
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'log_achievements_total',
]"""
ENRICHED_NUM = """NUM_FEATURES = [
    'days_since_release',
    'log_initial_price',
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'log_achievements_total',
    # Extra-signal features for the ordinal classifier (Step 4.5)
    'log_average_forever',
    'log_average_2weeks',
    'log_days_until_first_sale',
]"""

ORIG_CAT = """CAT_FEATURES = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
    'release_month',
    'player_engagement',
]"""
ENRICHED_CAT = """CAT_FEATURES = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
    'release_month',
    'player_engagement',
    'is_legacy',
]"""


# ---- Apply ---------------------------------------------------------------

set_code("step45_bins",       STEAM_BINS_SRC)
set_code("step45_plot",       STEAM_PLOT_SRC)
set_code("step45_total_norm", STEAM_TOTAL_NORM_SRC)
disc = get_cell("step45_disc")
if disc is not None:
    disc["source"] = STEAM_DISC_MD

# prep_features: revert NUM/CAT lists
pf = get_cell("prep_features")
if pf is not None:
    src = "".join(pf["source"])
    changed = False
    if ENRICHED_NUM in src:
        src = src.replace(ENRICHED_NUM, ORIG_NUM, 1)
        changed = True
    if ENRICHED_CAT in src:
        src = src.replace(ENRICHED_CAT, ORIG_CAT, 1)
        changed = True
    if changed:
        ast.parse(src)
        pf["source"]          = src.splitlines(keepends=True)
        pf["outputs"]         = []
        pf["execution_count"] = None
        print("prep_features: NUM/CAT_FEATURES reverted to original (without new features)")
    else:
        print("prep_features: anchors not found verbatim (already reverted?)")

# Sanity: setup_backfills was never modified (cell didn't exist), so nothing to undo there
print("setup_backfills: was already absent before the patch -- nothing to revert")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\n{NB.name} -- {len(nb['cells'])} cells, restored to Steam-tier state.")

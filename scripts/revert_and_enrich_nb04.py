"""Three changes to nb 04 (and nb 04c for the SMOTE variant):

  1. REVERT step45 bins to 20 even-width 5% bins.
  2. ENRICH features: add is_legacy, log_average_forever, log_average_2weeks,
     log_days_until_first_sale to NUM/CAT_FEATURES.
  3. BACKFILL the new features in setup_backfills so they exist on df even
     if cleaned_games predates the audit (defensive merge from raw tables).

The new features were chosen for *discount-tier discrimination*:

  - `is_legacy` (bool) -- legacy games >5y old have established sale cadence;
    helps separate "still on the new-release shelf" from "back-catalog."
  - `log_average_forever` (continuous) -- SteamSpy lifetime playtime average.
    Engaging games (high playtime) rarely deeply discount; throwaway games
    discount aggressively.
  - `log_average_2weeks` (continuous) -- recent engagement signal. A game
    still being actively played holds price; a dropped game discounts.
  - `log_days_until_first_sale` (continuous) -- direct strategy signal.
    Publishers who hold price for 6+ months tend to maintain higher floors;
    quick-discounters slide down faster.

Notebook 04c (SMOTE) gets the same patches so a re-run picks up the richer
feature space too.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_BASE  = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
NB_SMOTE = ROOT / "notebooks" / "04c_ml_part1_smote.ipynb"


# ---- Replacement sources ----------------------------------------------------

# 5%-bin revert -- step45_bins
NEW_BINS_SRC = '''# Build the 20-bin discount target. Each bin spans 5%.
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
    bar = "#" * max(1, int(n / counts.max() * 40))
    print(f'  bin {bin_id:>2} ({pct_low:>3.0f}-{pct_high:>3.0f}%):  {n:>4,}  {bar}')
'''

# 5%-bin step45_plot
NEW_PLOT_SRC = '''fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Confusion matrix (row-normalized) ---
cm = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                      normalize='true')
im_cm = axes[0].imshow(cm, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
cbar = plt.colorbar(im_cm, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Proportion within actual bin', fontsize=9)

# Tick at every other bin BOUNDARY (-0.5, 1.5, 3.5, ...) labeled with the
# actual % edge.
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
'''

# 5%-bin step45_total_norm
NEW_TOTAL_NORM_SRC = '''cm_total = confusion_matrix(y_test_ord, pred_ord, labels=list(range(N_BINS)),
                            normalize='all')

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm_total, cmap='Blues', aspect='auto', origin='lower',
               vmin=0, vmax=cm_total.max())
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Share of total test predictions', fontsize=9)

boundary_positions = np.arange(-0.5, N_BINS, 2)
boundary_labels    = [f'{int(BIN_EDGES[i]*100)}%' for i in range(0, N_BINS + 1, 2)]
ax.set_xticks(boundary_positions); ax.set_xticklabels(boundary_labels)
ax.set_yticks(boundary_positions); ax.set_yticklabels(boundary_labels)
ax.set_xlabel('Predicted discount bin (5% wide)')
ax.set_ylabel('Actual discount bin (5% wide)')
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
    actual_lbl = f'{int(BIN_EDGES[a]*100)}-{int(BIN_EDGES[a+1]*100)}%'
    pred_lbl   = f'{int(BIN_EDGES[p]*100)}-{int(BIN_EDGES[p+1]*100)}%'
    flag = ' <-- diagonal' if a == p else ''
    print(f'  actual {actual_lbl:>9}  ->  predicted {pred_lbl:>9}  : {share*100:>5.1f}%{flag}')
'''

NEW_DISC_SRC = [
    "### Ordinal classification - narrative\n",
    "\n",
    "20 even-width 5% bins on `max_discount_ever`. The classifier is supplemented with four extra features (vs the regression in Step 4.4) intended to give it discount-strategy signal:\n",
    "\n",
    "- `is_legacy` -- is the game >5 years old? (legacy stock has established cadence)\n",
    "- `log_average_forever` -- SteamSpy lifetime playtime average; engaging games hold price\n",
    "- `log_average_2weeks` -- recent engagement; dropped games discount\n",
    "- `log_days_until_first_sale` -- pricing-strategy signal (publishers who hold for 6+ months keep higher floors)\n",
    "\n",
    "**Headline metrics (compare to baseline):**\n",
    "\n",
    "- **Within +/- 1 bin** -- *\"the model lands within +/- 5pp of the actual deepest discount X% of the time.\"*\n",
    "- **MABE (pp)** -- average error, percentage points. With uniform 5% bins this is `mean_bin_distance * 5` and the units are unambiguous.\n",
    "\n",
    "**If the matrix still shows vertical-stripe collapse (model concentrating on a few popular bins):** the SMOTE variant in `04c_ml_part1_smote.ipynb` is the next lever to try. With these four extra features, SMOTE has more discriminating dimensions to interpolate within, so it should help more than it did the first time.\n",
]


# Setup-backfill insertion: append four backfills before the final print
BACKFILL_INSERT = '''
# --- Extra-feature backfills for the ordinal classifier (Step 4.5) -------
# These four features add discount-strategy signal that the regression
# features don't capture. Defensive: only computed if not already in df.

if 'is_legacy' not in df.columns:
    df['is_legacy'] = (df['days_since_release'] > 1825).astype(int)

if 'log_average_forever' not in df.columns:
    spy = pd.read_sql_query(
        'SELECT appid, average_forever, average_2weeks FROM steamspy', conn,
    )
    df = df.merge(spy, on='appid', how='left')
    df['average_forever']      = df['average_forever'].fillna(0).clip(lower=0)
    df['average_2weeks']       = df['average_2weeks'].fillna(0).clip(lower=0)
    df['log_average_forever']  = np.log1p(df['average_forever'])
    df['log_average_2weeks']   = np.log1p(df['average_2weeks'])

if 'log_days_until_first_sale' not in df.columns:
    first_sale = pd.read_sql_query("""
        SELECT appid, MIN(timestamp) AS first_sale_timestamp
        FROM price_history WHERE cut > 0 GROUP BY appid
    """, conn)
    first_sale['first_sale_dt'] = pd.to_datetime(first_sale['first_sale_timestamp'], utc=True)
    df = df.merge(first_sale[['appid', 'first_sale_dt']], on='appid', how='left')
    df['_release_dt'] = pd.to_datetime(df['release_date'], errors='coerce', utc=True)
    days_until = (df['first_sale_dt'] - df['_release_dt']).dt.days
    df['days_until_first_sale']     = days_until.fillna(-1).astype(int)
    df['log_days_until_first_sale'] = np.log1p(df['days_until_first_sale'].clip(lower=0))
    df = df.drop(columns=['_release_dt', 'first_sale_dt'])

print(f"  is_legacy:                  {df['is_legacy'].sum():,} of {len(df):,} games are legacy (>5y old)")
print(f"  log_average_forever:        median {df['log_average_forever'].median():.2f}  (raw median {df['average_forever'].median():.0f} min)")
print(f"  log_average_2weeks:         median {df['log_average_2weeks'].median():.2f}")
print(f"  log_days_until_first_sale:  median {df['log_days_until_first_sale'].median():.2f}  (raw median {df['days_until_first_sale'].median():.0f} days)")
'''


# Feature lists update (in prep_features cell)
NEW_NUM_FEATURES_BLOCK = """NUM_FEATURES = [
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
OLD_NUM_FEATURES_BLOCK = """NUM_FEATURES = [
    'days_since_release',
    'log_initial_price',
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'log_achievements_total',
]"""

NEW_CAT_FEATURES_BLOCK = """CAT_FEATURES = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
    'release_month',
    'player_engagement',
    'is_legacy',
]"""
OLD_CAT_FEATURES_BLOCK = """CAT_FEATURES = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
    'release_month',
    'player_engagement',
]"""


# ---- Patch helper ---------------------------------------------------------

def patch_notebook(nb_path: Path, label: str) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    get_cell = lambda cid: next((c for c in nb["cells"] if c.get("id") == cid), None)

    def set_code(cid, src):
        cell = get_cell(cid)
        if cell is None:
            print(f"  WARN  {label}: cell {cid} missing, skipped")
            return
        ast.parse(src)
        cell["source"]          = src.splitlines(keepends=True)
        cell["outputs"]         = []
        cell["execution_count"] = None

    # 1. Revert 5% bin source for the ordinal classifier
    set_code("step45_bins",       NEW_BINS_SRC)
    set_code("step45_plot",       NEW_PLOT_SRC)
    set_code("step45_total_norm", NEW_TOTAL_NORM_SRC)
    disc = get_cell("step45_disc")
    if disc is not None:
        disc["source"] = NEW_DISC_SRC

    # 2. Backfill new features in setup_backfills
    sb = get_cell("setup_backfills")
    if sb is not None:
        src = "".join(sb["source"])
        if "log_days_until_first_sale" not in src:
            # Insert before the final per-feature summary prints (find last print line)
            insertion = BACKFILL_INSERT
            if not src.rstrip().endswith("\n"):
                src += "\n"
            src = src.rstrip() + "\n" + insertion
            ast.parse(src)
            sb["source"]          = src.splitlines(keepends=True)
            sb["outputs"]         = []
            sb["execution_count"] = None
        else:
            print(f"  {label}: setup_backfills already has new features (skipped)")

    # 3. Update NUM_FEATURES / CAT_FEATURES in prep_features
    pf = get_cell("prep_features")
    if pf is not None:
        src = "".join(pf["source"])
        replaced = False
        if OLD_NUM_FEATURES_BLOCK in src:
            src = src.replace(OLD_NUM_FEATURES_BLOCK, NEW_NUM_FEATURES_BLOCK, 1)
            replaced = True
        if OLD_CAT_FEATURES_BLOCK in src:
            src = src.replace(OLD_CAT_FEATURES_BLOCK, NEW_CAT_FEATURES_BLOCK, 1)
            replaced = True
        if replaced:
            ast.parse(src)
            pf["source"]          = src.splitlines(keepends=True)
            pf["outputs"]         = []
            pf["execution_count"] = None
            print(f"  {label}: prep_features NUM/CAT_FEATURES updated")
        else:
            # idempotent on re-run
            if "log_average_forever" in src and "is_legacy" in src:
                print(f"  {label}: prep_features already enriched (skipped)")
            else:
                print(f"  WARN  {label}: prep_features anchors not found verbatim")

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {label}: step45 cells reverted to 5% bins; backfills + features wired in")


print("Patching nb 04 (baseline):")
patch_notebook(NB_BASE, "nb 04")
print()
print("Patching nb 04c (SMOTE):")
patch_notebook(NB_SMOTE, "nb 04c")
print("\nDone.")

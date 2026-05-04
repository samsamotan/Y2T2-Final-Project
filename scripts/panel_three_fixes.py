"""Apply three fixes to the panel section in Step 4.6:
A) Add publisher target-encoding to panel features
B) Add zoomed 0-3 year subplot alongside full 0-10 view
C) Plot real-game envelopes (5 closest games per cluster) as shaded bands

Also patches the panel feature-importance cell to include PUB_FEATURE in
its feature-name list (same bug as 04 main had earlier).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB04 = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
NB02 = ROOT / "notebooks" / "02_data_cleaning.ipynb"


def find_cell(nb, contains, cell_type=None):
    for c in nb["cells"]:
        if cell_type and c["cell_type"] != cell_type:
            continue
        if contains in "".join(c["source"]):
            return c
    return None


def set_source(cell, source):
    lines = source.rstrip("\n").split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]


# =====================================================================
# 02 — add publisher to panel-features merge so future re-runs include it
# =====================================================================
nb = json.loads(NB02.read_text(encoding="utf-8"))
cell = find_cell(nb, "panel_features = panel.merge")
if cell:
    src = "".join(cell["source"])
    if "'publisher'" not in src:
        # Add publisher to the merged column list
        old = "'is_multiplayer', 'has_controller_support', 'achievements_total',"
        new = "'is_multiplayer', 'has_controller_support', 'achievements_total',\n        'publisher',"
        if old in src:
            src = src.replace(old, new)
            set_source(cell, src)
            print("[02] panel_features merge: added 'publisher' column")
NB02.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


# =====================================================================
# 04 — Cell d9c7a96a: panel feature setup with publisher target encoding
# =====================================================================
nb = json.loads(NB04.read_text(encoding="utf-8"))

setup_cell = next(c for c in nb["cells"] if c.get("id") == "d9c7a96a")
new_setup = '''# Backfill publisher in panel data if cleaned_discount_panel was built before
# 02 included it in the merge. After a future 02 re-run the panel table will
# carry publisher directly and this is a no-op.
if 'publisher' not in panel.columns:
    pub = pd.read_sql_query('SELECT appid, publisher FROM games', conn)
    panel = panel.merge(pub, on='appid', how='left')
panel['publisher'] = panel['publisher'].fillna('Unknown').astype(str)

# Feature setup for panel regression
panel_num_features = [
    'age_year',                # the trajectory dimension
    'initial_price',
    'review_score',
    'log_ownership',
    'achievements_total',
]
panel_cat_features = [
    'primary_genre',
    'developer_tier',
    'price_tier',
    'is_multiplayer',
    'has_controller_support',
]
panel_pub_feature = ['publisher']   # high-cardinality, target-encoded

panel_clean = panel.dropna(
    subset=panel_num_features + panel_cat_features + panel_pub_feature + ['max_discount']
).copy()

X_panel  = panel_clean[panel_num_features + panel_cat_features + panel_pub_feature]
y_panel  = panel_clean['max_discount']
groups_panel = panel_clean['appid'].values

# Three-branch preprocessor: numeric scaling, one-hot, target-encoded publisher.
# Same pattern as Step 4.2 -- publisher is the literature\\'s dominant signal
# for discount behaviour and adding it here should differentiate archetype
# trajectories more than genre + dev_tier alone.
from sklearn.preprocessing import TargetEncoder
panel_preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                                                      panel_num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'),                                panel_cat_features),
    ('pub', TargetEncoder(smooth=10.0, target_type='continuous', random_state=42), panel_pub_feature),
])

print(f'Panel modelling rows         : {len(panel_clean):,}')
print(f'Unique games                 : {panel_clean["appid"].nunique():,}')
print(f'Features (num / cat / pub)   : {len(panel_num_features)} / {len(panel_cat_features)} / {len(panel_pub_feature)}')'''
set_source(setup_cell, new_setup)
print("[04] Cell d9c7a96a: added publisher target-encoding to panel preprocessor")


# =====================================================================
# 04 — Cell 02ef2e3c: archetypes with envelope bands + zoomed subplot
# =====================================================================
arch_cell = next(c for c in nb["cells"] if c.get("id") == "02ef2e3c")
new_arch = '''# Predicted depreciation curves -- data-driven archetypes via KMeans clustering.
#
# K-selection: sweep K from 2 to 8, pick K* = argmax(silhouette score).
# Each archetype = cluster centroid + modal categorical attributes + modal
# publisher. For each archetype we also find the 5 real games closest to
# the centroid in feature space and plot their predicted curves as a
# shaded envelope so the within-archetype variation is visible.

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

best_panel_model.fit(X_panel, y_panel)   # refit on all data for inference plots

ARCHETYPE_FEATURES = ['initial_price', 'review_score', 'log_ownership', 'achievements_total']
X_arch = df_clean[ARCHETYPE_FEATURES].values
arch_scaler = StandardScaler()
X_arch_scaled = arch_scaler.fit_transform(X_arch)

K_RANGE = list(range(2, 9))
inertias = []
silhouettes = []
for k in K_RANGE:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_arch_scaled)
    inertias.append(km_k.inertia_)
    silhouettes.append(silhouette_score(X_arch_scaled, km_k.labels_))

K_OPTIMAL = K_RANGE[int(np.argmax(silhouettes))]
print(f'K-selection sweep:')
for k, inert, sil in zip(K_RANGE, inertias, silhouettes):
    star = ' <-- selected' if k == K_OPTIMAL else ''
    print(f'  K={k}:  inertia={inert:>8.0f}  silhouette={sil:.3f}{star}')
print(f'\\nChosen K* = {K_OPTIMAL} (highest silhouette)')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(K_RANGE, inertias, marker='o', color=PALETTE['cyan'])
axes[0].axvline(K_OPTIMAL, color='red', linestyle='--', alpha=0.6, label=f'K*={K_OPTIMAL}')
axes[0].set_xlabel('K (number of clusters)')
axes[0].set_ylabel('Inertia (within-cluster sum of squares)')
axes[0].set_title('Elbow plot')
axes[0].legend()
axes[1].plot(K_RANGE, silhouettes, marker='s', color=PALETTE['orange'])
axes[1].axvline(K_OPTIMAL, color='red', linestyle='--', alpha=0.6, label=f'K*={K_OPTIMAL}')
axes[1].set_xlabel('K (number of clusters)')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score per K (higher = better separation)')
axes[1].legend()
plt.tight_layout()
plt.show()

# Final clustering with K*
final_km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10).fit(X_arch_scaled)
df_clean = df_clean.copy()
df_clean['archetype_cluster'] = final_km.labels_

# Build archetype profiles + find 5 closest real games per cluster
import numpy as np
archetypes = []
for cluster_id in range(K_OPTIMAL):
    members = df_clean[df_clean['archetype_cluster'] == cluster_id]
    centroid_num = members[ARCHETYPE_FEATURES].median()
    dominant_dev = members['developer_tier'].mode().iloc[0] if not members['developer_tier'].mode().empty else 'Unknown'
    dominant_price = members['price_tier'].mode().iloc[0] if not members['price_tier'].mode().empty else 'Unknown'
    dominant_genre = members['primary_genre'].mode().iloc[0] if not members['primary_genre'].mode().empty else 'Unknown'
    dominant_publisher = members['publisher'].mode().iloc[0] if not members['publisher'].mode().empty else 'Unknown'

    p = centroid_num['initial_price']
    if p < 5:    price_desc = 'Budget'
    elif p < 15: price_desc = 'Mid'
    elif p < 30: price_desc = 'Standard'
    else:        price_desc = 'Premium'
    mp_desc = 'Multiplayer' if (members['is_multiplayer'].mean() > 0.5) else 'Singleplayer'
    r = centroid_num['review_score']
    if r > 0.90:   rev_desc = 'highly-rated'
    elif r > 0.75: rev_desc = 'well-rated'
    else:          rev_desc = 'mixed-rated'

    # Find the 5 real games closest to this cluster\\'s centroid (Euclidean
    # in the scaled feature space)
    centroid_scaled = arch_scaler.transform([members[ARCHETYPE_FEATURES].median().values])[0]
    member_indices = members.index
    member_scaled = arch_scaler.transform(members[ARCHETYPE_FEATURES].values)
    distances = np.linalg.norm(member_scaled - centroid_scaled, axis=1)
    closest_5_idx = member_indices[np.argsort(distances)[:5]]
    closest_5 = df_clean.loc[closest_5_idx]

    archetypes.append({
        'cluster_id':             cluster_id,
        'label':                  f'{price_desc} {mp_desc} {dominant_genre} '
                                  f'(${centroid_num["initial_price"]:.0f}, {rev_desc}, n={len(members):,})',
        'initial_price':          centroid_num['initial_price'],
        'review_score':           centroid_num['review_score'],
        'log_ownership':          centroid_num['log_ownership'],
        'achievements_total':     centroid_num['achievements_total'],
        'primary_genre':          dominant_genre,
        'developer_tier':         dominant_dev,
        'price_tier':             dominant_price,
        'is_multiplayer':         int(members['is_multiplayer'].mean() > 0.5),
        'has_controller_support': int(members['has_controller_support'].mean() > 0.5),
        'publisher':              dominant_publisher,
        'cluster_size':           len(members),
        'closest_5_games':        closest_5,
    })

print(f'\\nDerived {len(archetypes)} archetypes from K-means clusters:')
for a in archetypes:
    print(f"  - {a['label']}")
    print(f"      modal publisher: {a['publisher'][:60]}")

# Predict for centroid AND for the 5 closest real games per cluster.
# Centroid line + min/max envelope over the 5 closest = visible variation.
ages = np.arange(0, 11)
panel_input_cols = panel_num_features + panel_cat_features + panel_pub_feature

def predict_curve_for_row(row_dict, ages_to_predict):
    grid = pd.DataFrame([{**row_dict, 'age_year': y} for y in ages_to_predict])
    return best_panel_model.predict(grid[panel_input_cols])

# Two side-by-side plots: full 0-10 view and zoomed 0-3 view.
# The differentiation is concentrated in the early years; the zoom makes
# that visible while the full view shows the convergence story.
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
import matplotlib.cm as cm
colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(archetypes)))

for a, color in zip(archetypes, colors):
    centroid_row = {
        'initial_price':          a['initial_price'],
        'review_score':           a['review_score'],
        'log_ownership':          a['log_ownership'],
        'achievements_total':     a['achievements_total'],
        'primary_genre':          a['primary_genre'],
        'developer_tier':         a['developer_tier'],
        'price_tier':             a['price_tier'],
        'is_multiplayer':         a['is_multiplayer'],
        'has_controller_support': a['has_controller_support'],
        'publisher':              a['publisher'],
    }
    centroid_pred = predict_curve_for_row(centroid_row, ages)

    # Predict each of the 5 closest real games -- gives an envelope
    member_preds = []
    for _, game in a['closest_5_games'].iterrows():
        game_row = {col: game[col] for col in ARCHETYPE_FEATURES}
        for col in ['primary_genre', 'developer_tier', 'price_tier',
                    'is_multiplayer', 'has_controller_support', 'publisher']:
            game_row[col] = game[col]
        member_preds.append(predict_curve_for_row(game_row, ages))
    member_preds = np.array(member_preds)
    lo = member_preds.min(axis=0)
    hi = member_preds.max(axis=0)

    for ax, x_max in [(ax_full, 10), (ax_zoom, 3)]:
        mask_age = ages <= x_max
        # Shaded envelope: range across 5 closest real games
        ax.fill_between(ages[mask_age], lo[mask_age], hi[mask_age],
                        color=color, alpha=0.15)
        # Centroid line
        ax.plot(ages[mask_age], centroid_pred[mask_age],
                marker='o', linewidth=2, color=color, label=a['label'])

ax_full.set_title('Full lifetime view (0-10 years)')
ax_full.set_xlabel('Game age (years)')
ax_full.set_ylabel('Predicted peak discount %')
ax_full.set_ylim(0, 100)
ax_full.set_xlim(0, 10)

ax_zoom.set_title('Early-life zoom (0-3 years -- where archetypes differ)')
ax_zoom.set_xlabel('Game age (years)')
ax_zoom.set_ylim(0, 100)
ax_zoom.set_xlim(0, 3)

# Single shared legend below
fig.legend(*ax_full.get_legend_handles_labels(), loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(paths.outputs_dir / 'panel_depreciation_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nSaved to {paths.outputs_dir / 'panel_depreciation_curves.png'}")
print('Shaded bands = range across the 5 real games closest to each cluster centroid.')'''
set_source(arch_cell, new_arch)
print("[04] Cell 02ef2e3c: added publisher to archetypes + envelope bands + zoomed view")


# =====================================================================
# 04 — Cell 84846f23: panel feature importance (length mismatch fix)
# =====================================================================
fi_cell = next(c for c in nb["cells"] if c.get("id") == "84846f23")
fi_src = "".join(fi_cell["source"])
old_line = "panel_feature_names = panel_num_features + list(panel_ohe.get_feature_names_out(panel_cat_features))"
new_line = "panel_feature_names = panel_num_features + list(panel_ohe.get_feature_names_out(panel_cat_features)) + panel_pub_feature"
if old_line in fi_src:
    fi_src = fi_src.replace(old_line, new_line)
    set_source(fi_cell, fi_src)
    print("[04] Cell 84846f23: added panel_pub_feature to feat_names")


# Save
NB04.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"  -> wrote {NB04.name}")

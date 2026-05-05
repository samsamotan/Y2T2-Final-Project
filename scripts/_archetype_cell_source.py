# Predicted depreciation curves -- data-driven archetypes via TRAJECTORY clustering.
#
# Why trajectory clustering instead of raw-feature clustering:
#   Clustering on [price, review, ownership, achievements] is biased toward
#   the densest regions of feature space, which produced two redundant
#   "Mid Singleplayer Strategy" clusters and a degenerate 7-game corner.
#   We now cluster on the panel model's predicted curves: each game is
#   represented by its 11-year predicted max-discount trajectory. The
#   clusters are then defined by what we actually care about -- discount
#   behaviour -- rather than by accidents of input feature density.
#
# Workflow:
#   1. Refit best_panel_model on all panel data
#   2. Predict the 11-year curve for every unique game in df_clean
#   3. KMeans on the (n_games, 11) trajectory matrix; K* via inertia elbow
#   4. For each cluster: mean trajectory + P25/P75 envelope + modal attrs

from sklearn.cluster import KMeans
import matplotlib.cm as cm

best_panel_model.fit(X_panel, y_panel)   # refit on all panel data for inference

panel_input_cols = panel_num_features + panel_cat_features + panel_pub_feature

# 1. Build the (n_games, 11) trajectory matrix.
# Vectorised: stack 11 copies of the games table with age_year set 0..10,
# predict once, reshape. Much faster than predicting one game at a time.
TRAJ_AGES = np.arange(0, 11)
games_uniq = df_clean.drop_duplicates('appid').reset_index(drop=True)
n_games    = len(games_uniq)
inference_grid = pd.concat(
    [games_uniq.assign(age_year=age) for age in TRAJ_AGES],
    ignore_index=True,
)
preds = best_panel_model.predict(inference_grid[panel_input_cols])
# preds layout: [age0_g0..gN, age1_g0..gN, ..., age10_g0..gN]
traj_matrix = preds.reshape(len(TRAJ_AGES), n_games).T   # (n_games, 11)
print(f'Trajectory matrix: {traj_matrix.shape}  (one curve per unique game)')

# 2. K-selection sweep on the trajectory matrix.
# No StandardScaler needed -- all values are 0..100 (same scale by construction).
# K* selected via the ELBOW METHOD on the inertia curve. Inertia monotonically
# decreases with K, so we don't argmin/argmax it directly. Instead we use the
# kneedle-style perpendicular-distance heuristic: the elbow is the K whose
# (k, inertia) point sits furthest from the straight line joining the first
# and last sweep points. That's a programmatic, defensible reading of "the
# K where adding more clusters stops paying off."
K_RANGE = list(range(2, 9))
inertias = []
for k in K_RANGE:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(traj_matrix)
    inertias.append(km_k.inertia_)

# Elbow detection -- perpendicular distance from each point to the line
# connecting (K_min, I_max) and (K_max, I_min), measured in normalized units.
ks_arr  = np.array(K_RANGE, dtype=float)
ins_arr = np.array(inertias,  dtype=float)
ks_n  = (ks_arr  - ks_arr.min())  / (ks_arr.max()  - ks_arr.min())
ins_n = (ins_arr - ins_arr.min()) / (ins_arr.max() - ins_arr.min())
p1 = np.array([ks_n[0],  ins_n[0]])
p2 = np.array([ks_n[-1], ins_n[-1]])
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)
distances = np.abs(line_vec[0]*(p1[1] - ins_n) - (p1[0] - ks_n)*line_vec[1]) / line_len
K_OPTIMAL = K_RANGE[int(np.argmax(distances))]

print('\nK-selection sweep on predicted trajectories (inertia + elbow detection):')
for k, inert, dist in zip(K_RANGE, inertias, distances):
    star = ' <-- elbow' if k == K_OPTIMAL else ''
    print(f'  K={k}:  inertia={inert:>10.0f}  elbow_dist={dist:.3f}{star}')
print(f'\nChosen K* = {K_OPTIMAL} (elbow of the inertia curve)')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(K_RANGE, inertias, marker='o', color=PALETTE['cyan'], label='Inertia')
# Draw the reference line whose perpendicular distance defines the elbow
ax.plot([K_RANGE[0], K_RANGE[-1]], [inertias[0], inertias[-1]],
        linestyle=':', color=PALETTE['muted'], linewidth=1, label='Reference line')
ax.axvline(K_OPTIMAL, color='red', linestyle='--', alpha=0.6, label=f'Elbow K*={K_OPTIMAL}')
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Inertia (within-cluster sum of squares)')
ax.set_title('Elbow plot -- K selected by maximum perpendicular distance')
ax.legend()
plt.tight_layout()
plt.show()

# 3. Final clustering at K*. Attach labels back to games_uniq + df_clean
final_km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10).fit(traj_matrix)
games_uniq['archetype_cluster'] = final_km.labels_
df_clean = df_clean.copy()
if 'archetype_cluster' in df_clean.columns:
    df_clean = df_clean.drop(columns=['archetype_cluster'])
df_clean = df_clean.merge(
    games_uniq[['appid', 'archetype_cluster']],
    on='appid', how='left',
)

# 4. Build per-cluster archetype profile
archetypes = []
for cluster_id in range(K_OPTIMAL):
    mask = (games_uniq['archetype_cluster'] == cluster_id).values
    members  = games_uniq[mask]
    sub_traj = traj_matrix[mask]

    mean_curve = sub_traj.mean(axis=0)
    p25_curve  = np.percentile(sub_traj, 25, axis=0)
    p75_curve  = np.percentile(sub_traj, 75, axis=0)

    dominant_dev   = members['developer_tier'].mode().iloc[0]   if not members['developer_tier'].mode().empty   else 'Unknown'
    dominant_genre = members['primary_genre'].mode().iloc[0]    if not members['primary_genre'].mode().empty    else 'Unknown'
    dominant_pub   = members['publisher'].mode().iloc[0]        if not members['publisher'].mode().empty        else 'Unknown'

    median_price = members['initial_price'].median()
    mean_review  = members['review_score'].mean()
    if median_price < 5:    price_desc = 'Budget'
    elif median_price < 15: price_desc = 'Mid'
    elif median_price < 30: price_desc = 'Standard'
    else:                   price_desc = 'Premium'
    mp_desc = 'Multiplayer' if (members['is_multiplayer'].mean() > 0.5) else 'Singleplayer'
    if mean_review > 0.90:   rev_desc = 'highly-rated'
    elif mean_review > 0.75: rev_desc = 'well-rated'
    else:                    rev_desc = 'mixed-rated'

    label = (f'{price_desc} {mp_desc} {dominant_genre} '
             f'(${median_price:.0f}, {rev_desc}, n={len(members):,})')

    # 5 real games whose predicted curves are closest to the cluster mean
    distances = np.linalg.norm(sub_traj - mean_curve, axis=1)
    closest_5 = members.iloc[np.argsort(distances)[:5]]

    archetypes.append({
        'cluster_id':       cluster_id,
        'label':            label,
        'mean_curve':       mean_curve,
        'p25_curve':        p25_curve,
        'p75_curve':        p75_curve,
        'cluster_size':     len(members),
        'developer_tier':   dominant_dev,
        'primary_genre':    dominant_genre,
        'publisher':        dominant_pub,
        'closest_5_games':  closest_5,
    })

print(f'\nDerived {len(archetypes)} archetypes from trajectory clusters:')
for a in archetypes:
    print(f'  - {a["label"]}')
    print(f'      modal publisher: {a["publisher"][:60]}')
    print(f'      curve: {a["mean_curve"][0]:.0f}% off at age 0  ->  '
          f'{a["mean_curve"][-1]:.0f}% off at age 10')
    titles = a['closest_5_games']['title'].tolist()[:3]
    print(f'      representative games: {", ".join(titles)}')

# 5. Plot archetype trajectories (full 0-10 + zoom 0-3) with P25/P75 envelopes
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(archetypes)))

for a, color in zip(archetypes, colors):
    for ax, x_max in [(ax_full, 10), (ax_zoom, 3)]:
        mask_age = TRAJ_AGES <= x_max
        ax.fill_between(TRAJ_AGES[mask_age],
                        a['p25_curve'][mask_age], a['p75_curve'][mask_age],
                        color=color, alpha=0.15)
        ax.plot(TRAJ_AGES[mask_age], a['mean_curve'][mask_age],
                marker='o', linewidth=2, color=color, label=a['label'])

ax_full.set_title('Full lifetime view (0-10 years)')
ax_full.set_xlabel('Game age (years)'); ax_full.set_ylabel('Predicted peak discount %')
ax_full.set_ylim(0, 100); ax_full.set_xlim(0, 10)

ax_zoom.set_title('Early-life zoom (0-3 years -- where archetypes differ)')
ax_zoom.set_xlabel('Game age (years)'); ax_zoom.set_ylim(0, 100); ax_zoom.set_xlim(0, 3)

fig.legend(*ax_full.get_legend_handles_labels(), loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(paths.outputs_dir / 'panel_depreciation_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved to {paths.outputs_dir / "panel_depreciation_curves.png"}')
print('Shaded bands = within-cluster P25-P75 of predicted trajectories.')

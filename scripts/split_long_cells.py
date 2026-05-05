"""Split the long code cells in notebook 04 into more digestible sections.

Targets (lines before -> total lines after split):
  - e6834f2c (99) -> 3 cells: imports / bootstrap / backfills
  - ccb410d0 (41) -> 2 cells: feature lists + filter funnel / preprocessor
  - aa53a665 (54) -> 2 cells: helper + fit / display + dummy check
  - ceb677e1 (39) -> 2 cells: tune / feat importance
  - d9c7a96a (50) -> 2 cells: backfill + features / preprocessor
  - 08a4cf2a (45) -> 2 cells: helper + fit / display
  - 02ef2e3c (177) -> 4 cells: matrix / K-selection / clustering / plot
  - dddfb73b (116) -> 2 cells: predict_buy_time / plot_buy_recommendation

Also fixes a latent bug in 02ef2e3c: a stale `dominant_dev` reference left
over from the earlier developer_tier removal.

Compile-checks every new cell before writing.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB   = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"


def split_cell(nb, target_id, parts):
    """Replace cell with target_id with N new cells.
    `parts` is a list of (new_id, source_string) tuples.
    """
    target_idx = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == target_id), None)
    if target_idx is None:
        raise SystemExit(f"cell {target_id} not found")

    new_cells = []
    for new_id, src in parts:
        ast.parse(src)   # raises SyntaxError if any split is broken
        new_cells.append({
            "cell_type": "code",
            "id": new_id,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            "outputs": [],
            "execution_count": None,
        })

    nb["cells"] = nb["cells"][:target_idx] + new_cells + nb["cells"][target_idx + 1 :]
    return target_idx, len(new_cells)


nb = json.loads(NB.read_text(encoding="utf-8"))

# =============================================================================
# 1. e6834f2c (setup) -> 3 cells
# =============================================================================
setup_imports = '''import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
'''

setup_bootstrap = '''# Bootstrap: make src importable
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_setup import setup_notebook
from src.plot_style import apply_style, PALETTE, TIER_COLORS

conn, paths = setup_notebook()

# Load the analysis-ready table built by 02_data_cleaning
df = pd.read_sql_query('SELECT * FROM cleaned_games', conn)
print(f'Loaded {len(df):,} rows from cleaned_games')
'''

setup_backfills = '''# Backfill columns that 04 needs but the current cleaned_games may not provide.
# After re-running 02 these will be present and these merges become no-ops.
#
#   ever_discounted    -- derived from raw price_history (1 if game ever had cut > 0)
#   achievements_total -- pulled from raw games table
#   max_discount_ever  -- deepest cut percent observed across the game's full price history
#   publisher          -- pulled from raw games table
if 'ever_discounted' not in df.columns:
    ever_disc = pd.read_sql_query(
        'SELECT appid, MAX(CASE WHEN cut > 0 THEN 1 ELSE 0 END) AS ever_discounted '
        'FROM price_history GROUP BY appid', conn,
    )
    df = df.merge(ever_disc, on='appid', how='left')

if 'achievements_total' not in df.columns:
    ach = pd.read_sql_query('SELECT appid, achievements_total FROM games', conn)
    df = df.merge(ach, on='appid', how='left')

if 'max_discount_ever' not in df.columns:
    mde = pd.read_sql_query(
        'SELECT appid, MAX(cut) AS max_discount_ever '
        'FROM price_history GROUP BY appid', conn,
    )
    df = df.merge(mde, on='appid', how='left')

if 'publisher' not in df.columns:
    pub = pd.read_sql_query('SELECT appid, publisher FROM games', conn)
    df = df.merge(pub, on='appid', how='left')

df['ever_discounted']    = df['ever_discounted'].fillna(0).astype(int)
df['achievements_total'] = df['achievements_total'].fillna(0).astype(int)
df['max_discount_ever']  = df['max_discount_ever'].fillna(0).astype(float)
df['publisher']          = df['publisher'].fillna('Unknown').astype(str)

print(f"  ever_discounted:    {df['ever_discounted'].sum():,} of {len(df):,} games have been on sale at least once")
print(f"  achievements_total: median {df['achievements_total'].median():.0f}, max {df['achievements_total'].max():,}")
print(f"  max_discount_ever:  median {df['max_discount_ever'].median():.0f}%, max {df['max_discount_ever'].max():.0f}%")
print(f"  publishers:         {df['publisher'].nunique():,} unique values (target-encoded as a feature)")
'''

idx, n = split_cell(nb, "e6834f2c", [
    ("setup_imports",   setup_imports),
    ("setup_bootstrap", setup_bootstrap),
    ("setup_backfills", setup_backfills),
])
print(f"e6834f2c -> 3 cells inserted at idx {idx}")

# =============================================================================
# 2. ccb410d0 (preprocessor) -> 2 cells
# =============================================================================
prep_features = '''# Feature lists. NUM_FEATURES are scaled, CAT_FEATURES are one-hot encoded,
# PUB_FEATURE is target-encoded separately because publisher has thousands
# of unique values (high cardinality).
NUM_FEATURES = [
    'days_since_release',
    'log_initial_price',
    'review_score',
    'log_total_reviews',
    'log_ownership',
    'log_achievements_total',
]
CAT_FEATURES = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
    'release_month',
    'player_engagement',
]
PUB_FEATURE = ['publisher']

# Filter funnel - explicit row-count diagnostic so we know exactly what was dropped.
required = NUM_FEATURES + CAT_FEATURES + PUB_FEATURE + ['max_discount_ever']
print('Filter funnel:')
print(f'  Loaded from cleaned_games                : {len(df):>5,}')
print(f'  + initial_price > 0                       : {len(df_model):>5,}')
mask_features = df_model[required].notna().all(axis=1)
print(f'  + non-null required features and targets : {mask_features.sum():>5,}')
mask_observed = df_model['max_discount_ever'] > 0   # drop games never observed on sale (likely no ITAD coverage)
print(f'  + has observed sale history (cut > 0)    : {(mask_features & mask_observed).sum():>5,}')
df_clean = df_model[mask_features & mask_observed].copy()
print(f'\\nFinal modelling dataset: {len(df_clean):,} games')
'''

prep_pipeline = '''# Preprocessing pipeline -- three branches:
#   - StandardScaler on continuous numerics
#   - OneHotEncoder on low-cardinality categoricals (drop one column for binary
#     features so we don't carry redundant dummy pairs)
#   - TargetEncoder on `publisher` (high-cardinality; uses out-of-fold means
#     during fit so no leakage into the regression target)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                                                     NUM_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'),             CAT_FEATURES),
    ('pub', TargetEncoder(smooth=10.0, target_type='auto', random_state=42),      PUB_FEATURE),
])
X_transformed = preprocessor.fit_transform(
    df_clean[NUM_FEATURES + CAT_FEATURES + PUB_FEATURE],
    df_clean['max_discount_ever'] / 100,
)
print(f'Transformed feature space: {X_transformed.shape[1]} columns after one-hot + target encoding')
'''

idx, n = split_cell(nb, "ccb410d0", [
    ("prep_features", prep_features),
    ("prep_pipeline", prep_pipeline),
])
print(f"ccb410d0 -> 2 cells inserted at idx {idx}")

# =============================================================================
# 3. aa53a665 (regression bake-off) -> 2 cells
# =============================================================================
bake_models = '''def evaluate_regression(name, y_true, y_pred):
    return {
        'Model': name,
        'MAE':   mean_absolute_error(y_true, y_pred),
        'RMSE':  np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
    }

def fit_predict(name, model):
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train_reg, y_train_reg)
    return pipe, pipe.predict(X_test_reg)

regression_results = []
predictions = {}

# 1. Honest baseline -- predicts the mean for everything.
pipe, pred = fit_predict('Dummy (predicts mean)', DummyRegressor(strategy='mean'))
regression_results.append(evaluate_regression('Dummy (predicts mean)', y_test_reg, pred))
predictions['Dummy (predicts mean)'] = pred

# 2. Linear baseline
pipe, pred = fit_predict('Linear Regression', LinearRegression())
regression_results.append(evaluate_regression('Linear Regression', y_test_reg, pred))
predictions['Linear Regression'] = pred

# 3. Ridge -- regularized linear
pipe, pred = fit_predict('Ridge Regression', Ridge(alpha=1.0))
regression_results.append(evaluate_regression('Ridge Regression', y_test_reg, pred))
predictions['Ridge Regression'] = pred

# 4. Random Forest
pipe_rf, pred = fit_predict('Random Forest',
                            RandomForestRegressor(n_estimators=200, max_depth=12,
                                                  random_state=42, n_jobs=-1))
regression_results.append(evaluate_regression('Random Forest', y_test_reg, pred))
predictions['Random Forest'] = pred

# 5. Gradient Boosting
pipe_gb, pred = fit_predict('Gradient Boosting',
                            GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                      max_depth=4, random_state=42))
regression_results.append(evaluate_regression('Gradient Boosting', y_test_reg, pred))
predictions['Gradient Boosting'] = pred
'''

bake_summary = '''regression_results_df = pd.DataFrame(regression_results).sort_values('RMSE').reset_index(drop=True)
print('=== Regression bake-off - predicting max_discount_ever / 100 ===')
display(regression_results_df.round(4))

# Sanity check: did any real model beat Dummy?
dummy_rmse     = next(r['RMSE'] for r in regression_results if r['Model'].startswith('Dummy'))
best_real_rmse = min(r['RMSE']  for r in regression_results if not r['Model'].startswith('Dummy'))
margin = (dummy_rmse - best_real_rmse) / dummy_rmse * 100
print(f"\\nBest real model beats Dummy by {margin:.1f}% on RMSE.")
if margin < 5:
    print("  WARNING: Less than 5% improvement -- features may have weak signal.")
else:
    print("  OK: Models are learning real signal from the features.")
'''

idx, n = split_cell(nb, "aa53a665", [
    ("bake_models",  bake_models),
    ("bake_summary", bake_summary),
])
print(f"aa53a665 -> 2 cells inserted at idx {idx}")

# =============================================================================
# 4. ceb677e1 (RF tuning + feat imp) -> 2 cells
# =============================================================================
rf_tune = '''# Hyperparameter tuning on the Random Forest
rf_param_grid = {
    'model__n_estimators':     [200, 400],
    'model__max_depth':        [8, 12, None],
    'model__min_samples_leaf': [1, 3, 5],
    'model__max_features':     ['sqrt', 'log2'],
}

rf_grid = GridSearchCV(
    Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    rf_param_grid,
    scoring='neg_root_mean_squared_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
)
rf_grid.fit(X_train_reg, y_train_reg)
best_rf = rf_grid.best_estimator_
print(f'Best RF parameters: {rf_grid.best_params_}')
print(f'Best CV RMSE      : {-rf_grid.best_score_:.4f}')
'''

rf_featimp = '''# Feature importance from the tuned RF
# Pull feature names directly from the fitted ColumnTransformer so we get
# the right count regardless of how each branch (numeric / one-hot / target)
# expands its inputs. Strip the 'num__' / 'cat__' / 'pub__' prefixes.
prep = best_rf.named_steps['prep']
feat_names  = [n.split('__', 1)[-1] for n in prep.get_feature_names_out()]
importances = best_rf.named_steps['model'].feature_importances_
fi = (pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .head(15))

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(fi['Feature'][::-1], fi['Importance'][::-1], color=PALETTE['cyan'])
ax.set_xlabel('Feature importance')
ax.set_title('Top 15 features - Tuned Random Forest (target: max_discount_ever)')
plt.tight_layout()
plt.show()

print('\\nTop 5 features driving discount depth prediction:')
for _, row in fi.head(5).iterrows():
    print(f"  {row['Feature']:35s}  {row['Importance']:.4f}")
'''

idx, n = split_cell(nb, "ceb677e1", [
    ("rf_tune",    rf_tune),
    ("rf_featimp", rf_featimp),
])
print(f"ceb677e1 -> 2 cells inserted at idx {idx}")

# =============================================================================
# 5. d9c7a96a (panel features) -> 2 cells
# =============================================================================
panel_features_setup = '''# Backfill publisher in panel data if cleaned_discount_panel was built before
# 02 included it. After a future 02 re-run this is a no-op.
if 'publisher' not in panel.columns:
    pub = pd.read_sql_query('SELECT appid, publisher FROM games', conn)
    panel = panel.merge(pub, on='appid', how='left')
panel['publisher'] = panel['publisher'].fillna('Unknown').astype(str)

# Log-transform skewed panel numerics so they match Step 4.2's game-level
# feature space. log1p handles zeros (free games, no-achievement games).
panel['log_initial_price']      = np.log1p(panel['initial_price'])
panel['log_achievements_total'] = np.log1p(panel['achievements_total'])

# Feature setup for panel regression
panel_num_features = [
    'age_year',                # the trajectory dimension
    'log_initial_price',
    'review_score',
    'log_ownership',
    'log_achievements_total',
]
panel_cat_features = [
    'primary_genre',
    'is_multiplayer',
    'has_controller_support',
]
panel_pub_feature = ['publisher']   # high-cardinality, target-encoded
'''

panel_pipeline = '''# Drop rows with any null required feature, build modelling matrices.
panel_clean = panel.dropna(
    subset=panel_num_features + panel_cat_features + panel_pub_feature + ['max_discount']
).copy()

X_panel      = panel_clean[panel_num_features + panel_cat_features + panel_pub_feature]
y_panel      = panel_clean['max_discount']
groups_panel = panel_clean['appid'].values

# Three-branch preprocessor: numeric scaling, one-hot, target-encoded publisher.
# target_type='continuous' avoids the multiclass detection that fires when the
# target is integer-valued with few unique values (panel max_discount is 0-100
# in 5pt steps, which auto-detection misreads as multiclass).
panel_preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                                                      panel_num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'),                                panel_cat_features),
    ('pub', TargetEncoder(smooth=10.0, target_type='continuous', random_state=42), panel_pub_feature),
])

print(f'Panel modelling rows         : {len(panel_clean):,}')
print(f'Unique games                 : {panel_clean["appid"].nunique():,}')
print(f'Features (num / cat / pub)   : {len(panel_num_features)} / {len(panel_cat_features)} / {len(panel_pub_feature)}')
'''

idx, n = split_cell(nb, "d9c7a96a", [
    ("panel_features_setup", panel_features_setup),
    ("panel_pipeline",       panel_pipeline),
])
print(f"d9c7a96a -> 2 cells inserted at idx {idx}")

# =============================================================================
# 6. 08a4cf2a (panel bake-off) -> 2 cells
# =============================================================================
panel_bake_models = '''# Bake-off: 4 models on the panel target + a DummyRegressor honest baseline
panel_results = []

def evaluate_panel_model(name, model, fit_X, fit_y, test_X, test_y):
    model.fit(fit_X, fit_y)
    pred = model.predict(test_X)
    panel_results.append({
        'Model': name,
        'MAE':   mean_absolute_error(test_y, pred),
        'RMSE':  np.sqrt(mean_squared_error(test_y, pred)),
        'R²': r2_score(test_y, pred),
    })
    return pred

evaluate_panel_model(
    'Dummy (predicts mean)',
    Pipeline([('prep', panel_preprocessor), ('model', DummyRegressor(strategy='mean'))]),
    X_panel_train, y_panel_train, X_panel_test, y_panel_test,
)
evaluate_panel_model(
    'Linear Regression',
    Pipeline([('prep', panel_preprocessor), ('model', LinearRegression())]),
    X_panel_train, y_panel_train, X_panel_test, y_panel_test,
)
evaluate_panel_model(
    'Ridge Regression',
    Pipeline([('prep', panel_preprocessor), ('model', Ridge(alpha=1.0))]),
    X_panel_train, y_panel_train, X_panel_test, y_panel_test,
)
evaluate_panel_model(
    'Random Forest',
    Pipeline([('prep', panel_preprocessor),
              ('model', RandomForestRegressor(n_estimators=200, max_depth=12,
                                              random_state=42, n_jobs=-1))]),
    X_panel_train, y_panel_train, X_panel_test, y_panel_test,
)
evaluate_panel_model(
    'Gradient Boosting',
    Pipeline([('prep', panel_preprocessor),
              ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                  max_depth=4, random_state=42))]),
    X_panel_train, y_panel_train, X_panel_test, y_panel_test,
)
'''

panel_bake_summary = '''panel_results_df = pd.DataFrame(panel_results).sort_values('RMSE').reset_index(drop=True)
print('=== Panel Regression Results - predicting max_discount per (game, age_year) ===')
display(panel_results_df.round(3))
'''

idx, n = split_cell(nb, "08a4cf2a", [
    ("panel_bake_models",  panel_bake_models),
    ("panel_bake_summary", panel_bake_summary),
])
print(f"08a4cf2a -> 2 cells inserted at idx {idx}")

# =============================================================================
# 7. 02ef2e3c (archetypes) -> 4 cells -- ALSO fixes stale dominant_dev ref
# =============================================================================
arch_traj = '''# Predicted depreciation curves -- data-driven archetypes via TRAJECTORY clustering.
# Each game is represented by its 11-year predicted max-discount curve, and we
# cluster on those curves directly. Clusters are then defined by what we
# actually care about (discount behaviour) rather than by accidents of input
# feature density.
from sklearn.cluster import KMeans
import matplotlib.cm as cm

best_panel_model.fit(X_panel, y_panel)   # refit on all panel data for inference

panel_input_cols = panel_num_features + panel_cat_features + panel_pub_feature

# Build the (n_games, 11) trajectory matrix.
# Vectorised: stack 11 copies of the games table with age_year set 0..10,
# predict once, reshape. Much faster than predicting one game at a time.
TRAJ_AGES  = np.arange(0, 11)
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
'''

arch_kselect = '''# K-selection sweep on the trajectory matrix. No StandardScaler needed --
# all values are 0..100 (same scale by construction).
#
# K* selected via the ELBOW METHOD on inertia. Since inertia monotonically
# decreases with K, we use the kneedle-style perpendicular-distance heuristic:
# the elbow is the K whose (k, inertia) point sits furthest from the straight
# line joining the first and last sweep points.
K_RANGE  = list(range(2, 9))
inertias = []
for k in K_RANGE:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(traj_matrix)
    inertias.append(km_k.inertia_)

# Perpendicular distance from each point to the line connecting endpoints,
# measured in normalized units.
ks_arr  = np.array(K_RANGE,  dtype=float)
ins_arr = np.array(inertias, dtype=float)
ks_n    = (ks_arr  - ks_arr.min())  / (ks_arr.max()  - ks_arr.min())
ins_n   = (ins_arr - ins_arr.min()) / (ins_arr.max() - ins_arr.min())
p1 = np.array([ks_n[0],  ins_n[0]])
p2 = np.array([ks_n[-1], ins_n[-1]])
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)
distances = np.abs(line_vec[0]*(p1[1] - ins_n) - (p1[0] - ks_n)*line_vec[1]) / line_len
K_OPTIMAL = K_RANGE[int(np.argmax(distances))]

print('K-selection sweep on predicted trajectories (inertia + elbow detection):')
for k, inert, dist in zip(K_RANGE, inertias, distances):
    star = ' <-- elbow' if k == K_OPTIMAL else ''
    print(f'  K={k}:  inertia={inert:>10.0f}  elbow_dist={dist:.3f}{star}')
print(f'\\nChosen K* = {K_OPTIMAL} (elbow of the inertia curve)')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(K_RANGE, inertias, marker='o', color=PALETTE['cyan'], label='Inertia')
ax.plot([K_RANGE[0], K_RANGE[-1]], [inertias[0], inertias[-1]],
        linestyle=':', color=PALETTE['muted'], linewidth=1, label='Reference line')
ax.axvline(K_OPTIMAL, color='red', linestyle='--', alpha=0.6, label=f'Elbow K*={K_OPTIMAL}')
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Inertia (within-cluster sum of squares)')
ax.set_title('Elbow plot -- K selected by maximum perpendicular distance')
ax.legend()
plt.tight_layout()
plt.show()
'''

arch_profiles = '''# Final clustering at K*. Attach labels back to games_uniq + df_clean.
final_km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10).fit(traj_matrix)
games_uniq['archetype_cluster'] = final_km.labels_
df_clean = df_clean.copy()
if 'archetype_cluster' in df_clean.columns:
    df_clean = df_clean.drop(columns=['archetype_cluster'])
df_clean = df_clean.merge(games_uniq[['appid', 'archetype_cluster']], on='appid', how='left')

# Build per-cluster archetype profile: mean curve + P25/P75 envelope + modal attrs.
archetypes = []
for cluster_id in range(K_OPTIMAL):
    mask     = (games_uniq['archetype_cluster'] == cluster_id).values
    members  = games_uniq[mask]
    sub_traj = traj_matrix[mask]

    mean_curve = sub_traj.mean(axis=0)
    p25_curve  = np.percentile(sub_traj, 25, axis=0)
    p75_curve  = np.percentile(sub_traj, 75, axis=0)

    dominant_genre = members['primary_genre'].mode().iloc[0] if not members['primary_genre'].mode().empty else 'Unknown'
    dominant_pub   = members['publisher'].mode().iloc[0]     if not members['publisher'].mode().empty     else 'Unknown'

    median_price = members['initial_price'].median()
    mean_review  = members['review_score'].mean()
    if   median_price <  5:  price_desc = 'Budget'
    elif median_price < 15:  price_desc = 'Mid'
    elif median_price < 30:  price_desc = 'Standard'
    else:                    price_desc = 'Premium'
    mp_desc = 'Multiplayer' if (members['is_multiplayer'].mean() > 0.5) else 'Singleplayer'
    if   mean_review > 0.90: rev_desc = 'highly-rated'
    elif mean_review > 0.75: rev_desc = 'well-rated'
    else:                    rev_desc = 'mixed-rated'

    label = (f'{price_desc} {mp_desc} {dominant_genre} '
             f'(${median_price:.0f}, {rev_desc}, n={len(members):,})')

    # 5 real games whose predicted curves are closest to the cluster mean.
    distances = np.linalg.norm(sub_traj - mean_curve, axis=1)
    closest_5 = members.iloc[np.argsort(distances)[:5]]

    archetypes.append({
        'cluster_id':       cluster_id,
        'label':            label,
        'mean_curve':       mean_curve,
        'p25_curve':        p25_curve,
        'p75_curve':        p75_curve,
        'cluster_size':     len(members),
        'primary_genre':    dominant_genre,
        'publisher':        dominant_pub,
        'closest_5_games':  closest_5,
    })

print(f'Derived {len(archetypes)} archetypes from trajectory clusters:')
for a in archetypes:
    print(f'  - {a["label"]}')
    print(f'      modal publisher: {a["publisher"][:60]}')
    print(f'      curve: {a["mean_curve"][0]:.0f}% off at age 0  ->  {a["mean_curve"][-1]:.0f}% off at age 10')
    titles = a['closest_5_games']['title'].tolist()[:3]
    print(f'      representative games: {", ".join(titles)}')
'''

arch_plot = '''# Plot archetype trajectories: full lifetime view + early-life zoom, with
# P25/P75 envelopes showing within-cluster variation.
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
ax_full.set_xlabel('Game age (years)')
ax_full.set_ylabel('Predicted peak discount %')
ax_full.set_ylim(0, 100); ax_full.set_xlim(0, 10)

ax_zoom.set_title('Early-life zoom (0-3 years -- where archetypes differ)')
ax_zoom.set_xlabel('Game age (years)')
ax_zoom.set_ylim(0, 100); ax_zoom.set_xlim(0, 3)

fig.legend(*ax_full.get_legend_handles_labels(), loc='lower center',
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(paths.outputs_dir / 'panel_depreciation_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\\nSaved to {paths.outputs_dir / "panel_depreciation_curves.png"}')
print('Shaded bands = within-cluster P25-P75 of predicted trajectories.')
'''

idx, n = split_cell(nb, "02ef2e3c", [
    ("arch_traj",     arch_traj),
    ("arch_kselect",  arch_kselect),
    ("arch_profiles", arch_profiles),
    ("arch_plot",     arch_plot),
])
print(f"02ef2e3c -> 4 cells inserted at idx {idx}  (also fixed stale dominant_dev reference)")

# =============================================================================
# 8. dddfb73b (buy-time recommender) -> 2 cells
# =============================================================================
buy_predict = '''def predict_buy_time(query, target_pct=50, max_wait=10,
                     df=None, model=None, panel_data=None):
    """Predict the optimal buy year for a specific game, with both the
    model's predicted trajectory and the game's actual observed yearly
    max-discounts (from the panel data) for comparison.

    Parameters
    ----------
    query : int or str
        appid (int) or title substring (case-insensitive) - the game to look up.
    target_pct : float
        The discount % the buyer wants to wait for (default 50%).
    max_wait : int
        Max years to consider waiting (default 10).
    df, model, panel_data
        Default to df_clean / best_panel_model / panel from the surrounding
        notebook scope. Override if you want to use a different source.
    """
    df = df_clean if df is None else df
    model = best_panel_model if model is None else model
    panel_data = panel if panel_data is None else panel_data

    # Find the game
    if isinstance(query, (int, np.integer)):
        matches = df[df['appid'] == int(query)]
        query_str = f'appid {query}'
    else:
        matches = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
        query_str = f'title containing {query!r}'
    if len(matches) == 0:
        raise ValueError(f'No game in df_clean matching {query_str}. '
                         f'(Game may exist but be filtered out by max_discount_ever > 0.)')
    game = matches.iloc[0]

    # Build static feature row (everything except age_year), predict across ages.
    feature_cols = panel_num_features + panel_cat_features + panel_pub_feature
    static = {col: game[col] for col in feature_cols if col != 'age_year'}
    ages = np.arange(0, max_wait + 1)
    grid = pd.DataFrame([{**static, 'age_year': age} for age in ages])
    predicted = model.predict(grid[feature_cols])

    # Pull the game's actual yearly max-discounts from the panel.
    actual = (panel_data[panel_data['appid'] == int(game['appid'])]
              [['age_year', 'max_discount']]
              .sort_values('age_year'))

    # First age that crosses the target
    crossings = np.where(predicted >= target_pct)[0]
    if len(crossings) > 0:
        rec_age  = int(ages[crossings[0]])
        rec_disc = float(predicted[crossings[0]])
        msg = (f'{game["title"]}: buy at year {rec_age} '
               f'(predicted {rec_disc:.0f}% off, target was {target_pct}%)')
    else:
        rec_age  = None
        rec_disc = float(predicted[-1])
        msg = (f'{game["title"]}: never reaches {target_pct}% within {max_wait} years. '
               f'Best predicted discount: {rec_disc:.0f}% at year {max_wait}.')

    return {
        'game':                game,
        'ages':                ages,
        'predicted_curve':     predicted,
        'actual_yearly_max':   actual,
        'recommended_age':     rec_age,
        'expected_discount':   rec_disc,
        'target_pct':          target_pct,
        'message':             msg,
    }
'''

buy_plot = '''def plot_buy_recommendation(result, ax=None):
    """Visualize a single buy-time recommendation:
      cyan line   = model's predicted yearly peak discount
      dark X-line = this game's actual observed yearly max (from panel)
    The recommended year is reported in the chart subtitle (and on the
    returned dict's 'recommended_age'), not drawn as a vertical guide.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ages   = result['ages']
    pred   = result['predicted_curve']
    actual = result['actual_yearly_max']

    # Predicted trajectory
    ax.plot(ages, pred, marker='o', linewidth=2, color=PALETTE['cyan'],
            label='Predicted peak discount')

    # Actual observed yearly max-discounts -- connected line with X markers,
    # parallel to the predicted curve.
    if len(actual) > 0:
        ax.plot(actual['age_year'], actual['max_discount'],
                marker='X', linewidth=2, markersize=10,
                color=PALETTE['ink'], markeredgecolor='white', markeredgewidth=1.5,
                label=f'Actual observed (n={len(actual)})', zorder=5)

    title = result['game']['title']
    ax.set_title(f'{title}\\n{result["message"]}', fontsize=10)
    ax.set_xlabel('Game age (years)')
    ax.set_ylabel('Peak discount %')
    ax.set_ylim(0, 100); ax.set_xlim(0, max(ages))
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    return ax


print('predict_buy_time + plot_buy_recommendation defined.')
'''

idx, n = split_cell(nb, "dddfb73b", [
    ("buy_predict", buy_predict),
    ("buy_plot",    buy_plot),
])
print(f"dddfb73b -> 2 cells inserted at idx {idx}")

# Final write
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nFinal notebook: {len(nb['cells'])} cells.")

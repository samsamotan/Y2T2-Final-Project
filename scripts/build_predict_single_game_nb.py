"""Build notebook 07: predict buy-time for a single game (live scrape).

Two changes:
  1. Add a `save_model_artifact` cell to notebook 04 right after `buy_plot`.
     It dumps best_panel_model + the panel feature lists to outputs/panel_model.joblib
     so notebook 07 can load them without re-training.
  2. Create notebook 07 that loads the artifact, exposes a `fetch_features_for_appid`
     helper that tries the local DB first and falls back to live Steam/SteamSpy/ITAD
     API calls, runs predict_buy_time, and plots the result.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB04 = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
NB07 = ROOT / "notebooks" / "07_predict_single_game.ipynb"


# =============================================================================
# 1. Add save cell to notebook 04
# =============================================================================
nb = json.loads(NB04.read_text(encoding="utf-8"))

save_src = '''# Persist the trained panel model + feature lists so notebook 07 (single-game
# live prediction) can load them without re-training. Saved as a joblib artifact
# alongside the panel-depreciation-curves PNG.
import joblib

artifact_path = paths.outputs_dir / 'panel_model.joblib'
joblib.dump(
    {
        'model':              best_panel_model,
        'panel_num_features': panel_num_features,
        'panel_cat_features': panel_cat_features,
        'panel_pub_feature':  panel_pub_feature,
    },
    artifact_path,
)
print(f'Saved panel model artifact to {artifact_path}')
'''
ast.parse(save_src)

save_cell = {
    "cell_type":       "code",
    "id":              "save_model_artifact",
    "metadata":        {},
    "source":          save_src.splitlines(keepends=True),
    "outputs":         [],
    "execution_count": None,
}

# Insert directly after buy_plot
target_idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == "buy_plot")
# Don't insert twice if rerun
already_there = any(c.get("id") == "save_model_artifact" for c in nb["cells"])
if not already_there:
    nb["cells"].insert(target_idx + 1, save_cell)
    NB04.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Inserted save_model_artifact cell into notebook 04 (idx {target_idx + 1}).")
else:
    print("save_model_artifact cell already present in notebook 04 -- skipped.")


# =============================================================================
# 2. Build notebook 07
# =============================================================================
title_md = [
    "# 07 - Predict Buy-Time for a Single Game (Live Scrape)\n",
    "\n",
    "Takes one Steam `appid` and produces a Step-4.7-style buy recommendation:\n",
    "predicted yearly peak-discount curve overlaid with the game's actual\n",
    "observed yearly max discounts. Works for **any game**, not just the\n",
    "ones already in `cleaned_games`.\n",
    "\n",
    "**How it works:**\n",
    "1. Loads the trained panel model from `outputs/panel_model.joblib` (created by notebook 04).\n",
    "2. Looks up the game in the local DB first. If not found, falls back to live API calls (Steam Storefront + SteamSpy + IsThereAnyDeal).\n",
    "3. Builds the feature row the panel model expects, sweeps `age_year` 0..10, and predicts the trajectory.\n",
    "4. Pulls the game's actual per-year max discount from the panel (or derives it live from ITAD history) and overlays it on the chart.\n",
]

setup_imports_src = '''import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Bootstrap: make src importable when launched from notebooks/
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_setup import setup_notebook
from src.plot_style import apply_style, PALETTE
from src.utils import load_keys
from src import steam_api, steamspy_api, itad_api

conn, paths = setup_notebook()
keys = load_keys()
print(f"ITAD key loaded: {bool(keys.get('itad'))}")
'''

load_model_src = '''# Load the panel model + feature lists saved by notebook 04.
artifact_path = paths.outputs_dir / 'panel_model.joblib'
if not artifact_path.exists():
    raise FileNotFoundError(
        f"Model artifact not found at {artifact_path}.\\n"
        "Run notebook 04 first (specifically the cells through `arch_traj` which "
        "fits best_panel_model on all panel data, plus `save_model_artifact` "
        "which dumps the model to disk)."
    )

artifact = joblib.load(artifact_path)
best_panel_model    = artifact['model']
panel_num_features  = artifact['panel_num_features']
panel_cat_features  = artifact['panel_cat_features']
panel_pub_feature   = artifact['panel_pub_feature']
panel_input_cols    = panel_num_features + panel_cat_features + panel_pub_feature

print(f'Loaded panel model from {artifact_path.name}')
print(f'  num features: {panel_num_features}')
print(f'  cat features: {panel_cat_features}')
print(f'  pub feature:  {panel_pub_feature}')
'''

fetch_helpers_src = '''def _features_from_db(appid):
    """Pull a single game's panel feature row from cleaned_games + the per-year
    actual max-discount series from cleaned_discount_panel. Returns
    (features_dict, actual_df) or None if the game isn't in the DB."""
    games_row = pd.read_sql_query(
        "SELECT * FROM cleaned_games WHERE appid = ?",
        conn, params=(int(appid),),
    )
    if len(games_row) == 0:
        return None
    row = games_row.iloc[0]

    # If older cleaned_games rows lack the new log columns, derive them on the fly.
    log_initial_price       = row.get('log_initial_price')
    log_achievements_total  = row.get('log_achievements_total')
    log_ownership           = row.get('log_ownership')
    if pd.isna(log_initial_price):
        log_initial_price = float(np.log1p(row.get('initial_price') or 0))
    if pd.isna(log_achievements_total):
        log_achievements_total = float(np.log1p(row.get('achievements_total') or 0))
    if pd.isna(log_ownership):
        log_ownership = float(np.log10((row.get('ownership_midpoint') or 1) + 1))

    feats = {
        'title':                  str(row['title']),
        'appid':                  int(row['appid']),
        'log_initial_price':      float(log_initial_price),
        'review_score':           float(row['review_score']),
        'log_ownership':          float(log_ownership),
        'log_achievements_total': float(log_achievements_total),
        'primary_genre':          str(row['primary_genre']),
        'is_multiplayer':         int(row['is_multiplayer']),
        'has_controller_support': int(row['has_controller_support']),
        'publisher':              str(row['publisher']),
    }

    actual = pd.read_sql_query(
        "SELECT age_year, max_discount FROM cleaned_discount_panel "
        "WHERE appid = ? ORDER BY age_year",
        conn, params=(int(appid),),
    )
    return feats, actual


def _parse_release_date(s):
    """Best-effort parse of Steam Storefront's release-date string."""
    if not s:
        return None
    for fmt in ('%b %d, %Y', '%d %b, %Y', '%Y-%m-%d', '%B %d, %Y'):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt), tz='UTC')
        except ValueError:
            continue
    return None


def _features_from_live_api(appid):
    """Fetch a single game from Steam Storefront + SteamSpy + reviews + ITAD,
    then build the same feature row + actual yearly-max series the panel model
    expects."""
    print(f'  [1/4] Steam Storefront ...')
    details = steam_api.fetch_app_details(int(appid))
    if not details or not details.get('success'):
        raise RuntimeError(f'Steam Storefront returned no usable data for appid {appid}')
    data = details['data']

    title         = data.get('name', f'appid {appid}')
    publisher     = (data.get('publishers') or ['Unknown'])[0]
    genres        = [g['description'] for g in (data.get('genres') or [])]
    primary_genre = genres[0] if genres else 'Unknown'
    categories    = [c['description'] for c in (data.get('categories') or [])]
    mp_keywords   = ('Multi-player', 'Multiplayer', 'Online PvP', 'Online Co-op', 'Co-op')
    is_multiplayer         = int(any(k in c for c in categories for k in mp_keywords))
    has_controller_support = int(bool(data.get('controller_support')))
    achievements_total     = int((data.get('achievements') or {}).get('total') or 0)
    initial_price_cents    = (data.get('price_overview') or {}).get('initial')
    initial_price          = (initial_price_cents or 0) / 100   # cc=ph by default; ITAD will override below
    release_str            = (data.get('release_date') or {}).get('date')
    release_dt             = _parse_release_date(release_str)

    print(f'  [2/4] SteamSpy ...')
    ss = steamspy_api.fetch_appdetails(int(appid)) or {}
    ow_min, ow_max = steamspy_api._parse_owners(ss.get('owners'))
    ownership_midpoint = ((ow_min or 0) + (ow_max or 0)) / 2 or 350_000.0   # SteamSpy band floor

    print(f'  [3/4] Steam reviews ...')
    rev = steam_api.fetch_review_summary(int(appid)) or {}
    pos = rev.get('total_positive') or 0
    neg = rev.get('total_negative') or 0
    total = pos + neg
    review_score  = (pos / total) if total > 0 else 0.5
    total_reviews = total

    print(f'  [4/4] IsThereAnyDeal ...')
    actual = pd.DataFrame(columns=['age_year', 'max_discount'])
    api_key = keys.get('itad')
    if api_key:
        lookup = itad_api.lookup_appid(api_key, int(appid))
        if lookup and lookup.get('id'):
            history = itad_api.fetch_price_history(api_key, lookup['id'])
            if history:
                rows = []
                for h in history:
                    deal = h.get('deal') or {}
                    rows.append({
                        'price':   (deal.get('price')   or {}).get('amount'),
                        'regular': (deal.get('regular') or {}).get('amount'),
                        'cut':     deal.get('cut'),
                        'timestamp': h.get('timestamp'),
                    })
                ph = pd.DataFrame(rows)
                ph['timestamp'] = pd.to_datetime(ph['timestamp'], utc=True)

                # ITAD-derived initial_price (USD, regular-of-latest)
                latest_regular = ph.sort_values('timestamp')['regular'].dropna().tail(1)
                if not latest_regular.empty:
                    initial_price = float(latest_regular.iloc[0])

                # Per-year max discount, if we have a parseable release date
                if release_dt is not None and len(ph) > 0:
                    ph['observation_age_days'] = (ph['timestamp'] - release_dt).dt.days
                    ph = ph[ph['observation_age_days'] >= 0]
                    ph['age_year'] = (ph['observation_age_days'] // 365).astype(int).clip(upper=10)
                    actual = (ph[ph['cut'].fillna(0) > 0]
                              .groupby('age_year')['cut'].max()
                              .reset_index()
                              .rename(columns={'cut': 'max_discount'})
                              .sort_values('age_year'))
        else:
            print('     (ITAD has no record of this appid; actual overlay will be empty)')
    else:
        print('     (No ITAD key set; skipping price-history fetch)')

    feats = {
        'title':                  title,
        'appid':                  int(appid),
        'log_initial_price':      float(np.log1p(max(initial_price, 0))),
        'review_score':           float(review_score),
        'log_ownership':          float(np.log10(ownership_midpoint + 1)),
        'log_achievements_total': float(np.log1p(achievements_total)),
        'primary_genre':          primary_genre,
        'is_multiplayer':         is_multiplayer,
        'has_controller_support': has_controller_support,
        'publisher':              publisher,
    }
    return feats, actual


def fetch_features_for_appid(appid, prefer_live=False):
    """Returns (features_dict, actual_yearly_max_df, source_str).
    If prefer_live=False (default), tries the local DB first and falls back to
    a live API fetch only when the game isn't cached. If True, always fetches live.
    """
    if not prefer_live:
        cached = _features_from_db(appid)
        if cached is not None:
            feats, actual = cached
            return feats, actual, 'cleaned_games'
    print(f'Live fetch for appid {appid}...')
    feats, actual = _features_from_live_api(appid)
    return feats, actual, 'live API'
'''

predict_src = '''def predict_buy_time(features, actual=None, target_pct=50, max_wait=10):
    """Run the loaded panel model across ages 0..max_wait for one game,
    surface the first age that crosses target_pct, return everything the
    plot helper needs."""
    if actual is None:
        actual = pd.DataFrame(columns=['age_year', 'max_discount'])

    ages = np.arange(0, max_wait + 1)
    grid = pd.DataFrame([{**features, 'age_year': age} for age in ages])
    predicted = best_panel_model.predict(grid[panel_input_cols])

    crossings = np.where(predicted >= target_pct)[0]
    if len(crossings) > 0:
        rec_age  = int(ages[crossings[0]])
        rec_disc = float(predicted[crossings[0]])
        msg = (f"{features['title']}: buy at year {rec_age} "
               f"(predicted {rec_disc:.0f}% off, target was {target_pct}%)")
    else:
        rec_age  = None
        rec_disc = float(predicted[-1])
        msg = (f"{features['title']}: never reaches {target_pct}% within {max_wait} years. "
               f"Best predicted: {rec_disc:.0f}% at year {max_wait}.")

    return {
        'features':            features,
        'ages':                ages,
        'predicted_curve':     predicted,
        'actual_yearly_max':   actual,
        'recommended_age':     rec_age,
        'expected_discount':   rec_disc,
        'target_pct':          target_pct,
        'message':             msg,
    }


def plot_buy_recommendation(result, ax=None):
    """Cyan line = predicted; dark X-line = actual observed yearly max (if any)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ages   = result['ages']
    pred   = result['predicted_curve']
    actual = result['actual_yearly_max']

    ax.plot(ages, pred, marker='o', linewidth=2, color=PALETTE['cyan'],
            label='Predicted peak discount')
    if len(actual) > 0:
        ax.plot(actual['age_year'], actual['max_discount'],
                marker='X', linewidth=2, markersize=10,
                color=PALETTE['ink'], markeredgecolor='white', markeredgewidth=1.5,
                label=f'Actual observed (n={len(actual)})', zorder=5)

    ax.set_title(f'{result["features"]["title"]}\\n{result["message"]}', fontsize=10)
    ax.set_xlabel('Game age (years)')
    ax.set_ylabel('Peak discount %')
    ax.set_ylim(0, 100); ax.set_xlim(0, max(ages))
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    return ax
'''

demo_src = '''# Demo. Edit `appid` and `target_pct` to test any Steam game.
#
#   - appids already in cleaned_games will hit the DB cache (fast, ~0.1s)
#   - appids not in cleaned_games will fall back to live API calls (~5-10s)
#   - Set prefer_live=True to force live fetch even for cached games
#
# A few well-known appids to try:
#     413150  Stardew Valley
#     1086940 Baldur's Gate 3
#     220     Half-Life 2
#     105600  Terraria
#     292030  The Witcher 3: Wild Hunt
#     1091500 Cyberpunk 2077

appid       = 413150        # Stardew Valley
target_pct  = 50            # buyer's target discount %
prefer_live = False         # set True to force live API calls

features, actual, source = fetch_features_for_appid(appid, prefer_live=prefer_live)
print(f'\\nResolved {features["title"]!r} via {source}')
print(f'  publisher: {features["publisher"]}')
print(f'  primary_genre: {features["primary_genre"]}')
print(f'  panel rows for actual overlay: {len(actual)}')

result = predict_buy_time(features, actual=actual, target_pct=target_pct, max_wait=10)
print(f'\\n{result["message"]}')

fig, ax = plt.subplots(figsize=(10, 5))
plot_buy_recommendation(result, ax=ax)
plt.tight_layout()
plt.show()
'''

cells = [
    {"cell_type": "markdown", "id": "title",          "metadata": {}, "source": title_md},
    {"cell_type": "markdown", "id": "setup_md",       "metadata": {}, "source": ["## Setup\n"]},
    {"cell_type": "code",     "id": "setup",          "metadata": {}, "source": setup_imports_src.splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "markdown", "id": "load_md",        "metadata": {}, "source": ["## Load saved panel model\n"]},
    {"cell_type": "code",     "id": "load_model",     "metadata": {}, "source": load_model_src.splitlines(keepends=True),    "outputs": [], "execution_count": None},
    {"cell_type": "markdown", "id": "fetch_md",       "metadata": {}, "source": ["## Helpers: feature fetch (DB-first, live-fallback)\n"]},
    {"cell_type": "code",     "id": "fetch_helpers",  "metadata": {}, "source": fetch_helpers_src.splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "markdown", "id": "predict_md",     "metadata": {}, "source": ["## Predict + plot helpers\n"]},
    {"cell_type": "code",     "id": "predict",        "metadata": {}, "source": predict_src.splitlines(keepends=True),       "outputs": [], "execution_count": None},
    {"cell_type": "markdown", "id": "demo_md",        "metadata": {}, "source": ["## Demo\n"]},
    {"cell_type": "code",     "id": "demo",           "metadata": {}, "source": demo_src.splitlines(keepends=True),          "outputs": [], "execution_count": None},
]

# Compile-check every code cell
for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))

nb_07 = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB07.write_text(json.dumps(nb_07, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote: {NB07.relative_to(ROOT)}  ({len(cells)} cells)")

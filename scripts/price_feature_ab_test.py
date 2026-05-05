"""A/B/C empirical test: does keeping both `initial_price` and `price_tier`
in the feature set actually help, or is it dead-weight redundancy?

Variants:
  - CURRENT:    both initial_price (numeric) and price_tier (one-hot)
  - A (raw):    initial_price only, drop price_tier
  - B (tier):   price_tier only, drop initial_price

Test on:
  1. Panel regression (Step 4.6) - GroupKFold R^2 across 5 folds
  2. Game-level regression (Step 4.4) - 80/20 random split (matches notebook)

The 'better' answer should beat current within noise; the 'worse' answer
should give us the cost of choosing it.
"""
from __future__ import annotations
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "steam.db"

con = sqlite3.connect(DB)
panel = pd.read_sql_query("SELECT * FROM cleaned_discount_panel", con)
df    = pd.read_sql_query("SELECT * FROM cleaned_games", con)
con.close()

# ---------- Panel feature lists ----------
PANEL_NUM_BOTH    = ["age_year", "initial_price", "review_score", "log_ownership", "achievements_total"]
PANEL_NUM_NOPRICE = ["age_year",                   "review_score", "log_ownership", "achievements_total"]
PANEL_CAT_BOTH    = ["primary_genre", "developer_tier", "price_tier", "is_multiplayer", "has_controller_support"]
PANEL_CAT_NOTIER  = ["primary_genre", "developer_tier",                "is_multiplayer", "has_controller_support"]
PANEL_PUB         = ["publisher"]

# Backfill publisher if needed (matches notebook cell d9c7a96a)
if "publisher" not in panel.columns:
    pub = pd.read_sql_query("SELECT appid, publisher FROM games", sqlite3.connect(DB))
    panel = panel.merge(pub, on="appid", how="left")
panel["publisher"] = panel["publisher"].fillna("Unknown").astype(str)

def panel_pipe(num_features, cat_features):
    prep = ColumnTransformer([
        ("num", StandardScaler(),                          num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),    cat_features),
        ("pub", TargetEncoder(smooth=10.0, target_type="auto", random_state=42), PANEL_PUB),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ])

PANEL_VARIANTS = {
    "CURRENT (price + tier)": (PANEL_NUM_BOTH,    PANEL_CAT_BOTH),
    "A: price only          ": (PANEL_NUM_BOTH,    PANEL_CAT_NOTIER),
    "B: tier only           ": (PANEL_NUM_NOPRICE, PANEL_CAT_BOTH),
}

print("=" * 78)
print("Panel regression  -  5-fold GroupKFold (groups = appid)")
print("=" * 78)
for name, (num_features, cat_features) in PANEL_VARIANTS.items():
    needed = num_features + cat_features + PANEL_PUB + ["max_discount", "appid"]
    p = panel.dropna(subset=needed).copy()
    X = p[num_features + cat_features + PANEL_PUB]
    y = p["max_discount"]
    g = p["appid"].values

    pipe = panel_pipe(num_features, cat_features)
    cv   = GroupKFold(n_splits=5)
    r2s  = []
    maes = []
    rmses = []
    for tr, te in cv.split(X, y, groups=g):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
    n_cols = pipe.named_steps["prep"].fit(X, y).transform(X.head(2)).shape[1]
    print(f"  {name}  R2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}   "
          f"MAE = {np.mean(maes):.2f}  RMSE = {np.mean(rmses):.2f}   "
          f"n_cols = {n_cols}")

# ---------- Game-level feature lists (Step 4.4) ----------
GAME_NUM_BOTH    = ["days_since_release", "initial_price", "review_score", "log_total_reviews", "log_ownership", "achievements_total"]
GAME_NUM_NOPRICE = ["days_since_release",                   "review_score", "log_total_reviews", "log_ownership", "achievements_total"]
GAME_CAT_BOTH    = ["primary_genre", "developer_tier", "price_tier", "is_multiplayer", "has_controller_support", "release_month", "player_engagement"]
GAME_CAT_NOTIER  = ["primary_genre", "developer_tier",                "is_multiplayer", "has_controller_support", "release_month", "player_engagement"]
GAME_PUB         = ["publisher"]

# Apply the same filter funnel as cell ccb410d0
df_model = df[df["initial_price"] > 0].copy()
# log_total_reviews is derived in 04 setup; recreate it here
df_model["log_total_reviews"] = np.log10(df_model["total_reviews"].clip(lower=0) + 1)

def game_pipe(num_features, cat_features):
    prep = ColumnTransformer([
        ("num", StandardScaler(),                          num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),    cat_features),
        ("pub", TargetEncoder(smooth=10.0, target_type="auto", random_state=42), GAME_PUB),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ])

GAME_VARIANTS = {
    "CURRENT (price + tier)": (GAME_NUM_BOTH,    GAME_CAT_BOTH),
    "A: price only          ": (GAME_NUM_BOTH,    GAME_CAT_NOTIER),
    "B: tier only           ": (GAME_NUM_NOPRICE, GAME_CAT_BOTH),
}

print()
print("=" * 78)
print("Game-level regression  -  80/20 random split (matches notebook 4.4)")
print("=" * 78)
for name, (num_features, cat_features) in GAME_VARIANTS.items():
    needed = num_features + cat_features + GAME_PUB + ["max_discount_ever"]
    d = df_model.dropna(subset=needed).copy()
    d = d[d["max_discount_ever"] > 0]
    X = d[num_features + cat_features + GAME_PUB]
    y = d["max_discount_ever"] / 100
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = game_pipe(num_features, cat_features)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    n_cols = pipe.named_steps["prep"].transform(Xte.head(2)).shape[1]
    print(f"  {name}  R2 = {r2_score(yte, pred):.3f}   "
          f"MAE = {mean_absolute_error(yte, pred):.4f}  "
          f"RMSE = {np.sqrt(mean_squared_error(yte, pred)):.4f}   "
          f"n_cols = {n_cols}")

# ---------- Bonus: is the relationship monotonic? ----------
print()
print("=" * 78)
print("Diagnostic: is max_discount_ever vs initial_price monotonic?")
print("=" * 78)
d = df_model[(df_model["initial_price"] > 0) & (df_model["max_discount_ever"].notna()) & (df_model["max_discount_ever"] > 0)].copy()
d["price_decile"] = pd.qcut(d["initial_price"], 10, duplicates="drop")
agg = d.groupby("price_decile", observed=True)["max_discount_ever"].agg(["mean", "median", "count"]).reset_index()
agg["price_decile"] = agg["price_decile"].astype(str)
print(agg.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

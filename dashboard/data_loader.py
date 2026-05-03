"""
data_loader.py
--------------
Connects to the cleaned Steam SQLite DB and returns preprocessed DataFrames.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

CUSTOM_DB_PATH = r"C:\Users\acer\Y2T2-Final-Project\cleaned data\steam (1)_cleaned 1.db"

_DB_CANDIDATES = [
    Path(CUSTOM_DB_PATH),
    Path(__file__).parent.parent / "cleaned data" / "steam (1)_cleaned 1.db",
    Path(__file__).parent.parent / "data" / "steam.db",
    Path(__file__).parent / "steam.db",
]
_DB_CANDIDATES = [p for p in _DB_CANDIDATES if p is not None]

USD_TO_PHP = 56.0

TIER_COLORS = {
    "Premium Hold":          "#00e5ff",
    "Standard Depreciation": "#a3e635",
    "Heavy Discount":        "#fb923c",
    "Permanent Bargain":     "#f87171",
}

TIER_ORDER = ["Premium Hold", "Standard Depreciation", "Heavy Discount", "Permanent Bargain"]


def _find_db():
    for p in _DB_CANDIDATES:
        if p.exists():
            return p
    return None


def _retention_tier(r):
    if r > 0.85:   return "Premium Hold"
    if r > 0.50:   return "Standard Depreciation"
    if r > 0.25:   return "Heavy Discount"
    return "Permanent Bargain"


def _dev_tier(dev):
    AAA_KEYWORDS = {"valve", "ea", "ubisoft", "activision", "2k", "bethesda",
                    "rockstar", "square enix", "capcom", "bandai namco", "sega",
                    "sony", "microsoft", "nintendo", "warner", "thq"}
    if pd.isna(dev):
        return "Indie"
    d = str(dev).lower()
    if any(a in d for a in AAA_KEYWORDS):
        return "AAA"
    return "Mid"


def _make_synthetic():
    rng = np.random.default_rng(42)
    n = 800
    genres = ["Action", "RPG", "Strategy", "Indie", "Simulation",
              "Adventure", "Sports", "Racing", "Puzzle", "Horror"]
    dev_tiers = ["Indie", "Mid", "AAA"]

    days        = rng.integers(180, 4500, n).astype(float)
    initial_usd = rng.choice([5, 10, 15, 20, 30, 40, 60], n)
    initial_php = initial_usd * USD_TO_PHP
    age_factor  = np.exp(-days / 3500)
    review_sc   = rng.uniform(30, 100, n)
    noise       = rng.normal(0, 0.12, n)
    retention   = np.clip(age_factor + (review_sc - 50) / 500 + noise, 0.05, 1.0)
    current_php = initial_php * retention
    discount    = 1 - retention
    owners      = rng.integers(500, 5_000_000, n)

    release_dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 3500, n), unit="D"
    )

    df = pd.DataFrame({
        "appid":              rng.integers(100_000, 999_999, n),
        "title":              [f"Sample Game #{i+1}" for i in range(n)],
        "primary_genre":      rng.choice(genres, n),
        "developer":          rng.choice(["Indie Dev", "Mid Studio", "EA", "Valve"], n),
        "developer_tier":     rng.choice(dev_tiers, n, p=[0.60, 0.30, 0.10]),
        "initial_price_php":  initial_php,
        "current_price_php":  current_php,
        "discount_depth":     discount,
        "price_retention":    retention,
        "days_since_release": days,
        "release_date":       release_dates,
        "review_score":       review_sc,
        "total_reviews":      rng.integers(10, 80_000, n),
        "owners":             owners,
        "log_ownership":      np.log10(owners),
        "achievements_total": rng.integers(0, 200, n),
        "is_multiplayer":     rng.integers(0, 2, n),
        "metacritic_score":   rng.choice([np.nan, *range(40, 95)], n),
    })

    df["value_retention_tier"] = df["price_retention"].apply(_retention_tier)
    df["price_tier"] = pd.cut(
        df["initial_price_php"],
        bins=[0, 200, 1500, 1e9],
        labels=["Budget", "Mid", "Premium"]
    )
    return df


def _load_from_db(db_path):
    conn = sqlite3.connect(db_path)

    # Check row count first — skip empty DBs
    count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    if count == 0:
        conn.close()
        return None

    games_q = """
        SELECT
            g.appid, g.title, g.developer, g.publisher,
            g.release_date,
            g.launch_price_cents,
            g.current_price_cents,
            g.discount_percent,
            g.metacritic_score,
            g.achievements_total,
            g.controller_support,
            r.total_reviews,
            r.total_positive,
            r.total_negative,
            r.review_score,
            s.owners_min,
            s.owners_max,
            s.average_forever,
            s.average_2weeks
        FROM games g
        LEFT JOIN reviews_summary r ON g.appid = r.appid
        LEFT JOIN steamspy s        ON g.appid = s.appid
        WHERE g.type = 'game'
          AND g.is_free = 0
          AND g.launch_price_cents > 0
          AND g.current_price_cents IS NOT NULL
    """
    df = pd.read_sql(games_q, conn)

    try:
        gdf = pd.read_sql("SELECT appid, genre FROM game_genres", conn)
        pg  = gdf.groupby("appid")["genre"].first().reset_index()
        pg.columns = ["appid", "primary_genre"]
        df = df.merge(pg, on="appid", how="left")
    except Exception:
        df["primary_genre"] = "Unknown"

    conn.close()

    df["initial_price_php"] = df["launch_price_cents"]  / 100.0
    df["current_price_php"] = df["current_price_cents"] / 100.0
    df["price_retention"]   = (df["current_price_php"] / df["initial_price_php"]).clip(0, 1)
    df["discount_depth"]    = 1 - df["price_retention"]

    df["release_date"]       = pd.to_datetime(df["release_date"], errors="coerce")
    df["days_since_release"] = (pd.Timestamp.today() - df["release_date"]).dt.days.astype(float)

    df["owners"]             = ((df["owners_min"].fillna(0) + df["owners_max"].fillna(0)) / 2).astype(int)
    df["log_ownership"]      = np.log10(df["owners"].replace(0, 1))
    df["review_score"]       = df["review_score"].fillna(50)
    df["total_reviews"]      = df["total_reviews"].fillna(0).astype(int)
    df["achievements_total"] = df["achievements_total"].fillna(0).astype(int)
    df["primary_genre"]      = df["primary_genre"].fillna("Other")

    df["developer_tier"]       = df["developer"].apply(_dev_tier)
    df["value_retention_tier"] = df["price_retention"].apply(_retention_tier)
    df["price_tier"] = pd.cut(
        df["initial_price_php"],
        bins=[0, 200, 1500, 1e9],
        labels=["Budget", "Mid", "Premium"]
    )

    return df.dropna(subset=["initial_price_php", "current_price_php", "days_since_release"])


@st.cache_data(show_spinner=False)
def load_games():
    """
    Returns (df, is_real).
    is_real=True  → loaded from your actual cleaned DB
    is_real=False → synthetic demo data
    """
    db_path = _find_db()
    if db_path:
        try:
            df = _load_from_db(db_path)
            if df is not None and len(df) > 0:
                return df, True
        except Exception as e:
            st.sidebar.error(f"DB Error: {e}")

    return _make_synthetic(), False
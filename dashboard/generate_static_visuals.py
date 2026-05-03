"""
generate_static_visuals.py — Step 6.2
Generates all static visualizations for the technical report.
"""

import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ── Configuration & Paths ──────────────────────────────────────────────────────
# Prioritizing local data/steam.db, then fallbacks to known student paths
DB_PATHS = [
    Path("data/steam.db"),
    Path(r"C:/Users/Mico Uy/Downloads/Year 2 Term 2/Cleaned Data/steam (1)_cleaned 1.db"),
    Path(r"C:/Users/acer/Downloads/Year 2 Term 2/Cleaned Data/steam (1)_cleaned 1.db")
]

DB_PATH = next((p for p in DB_PATHS if p.exists()), Path("data/steam.db"))
OUT_DIR = Path("outputs/visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling (The "Cyber-Dark" Theme) ──────────────────────────────────────────
DARK_BG   = "#0a0a0a"
PANEL_BG  = "#111111"
TEXT_COL  = "#cccccc"
GRID_COL  = "#1e1e1e"
ACCENT    = "#00e5ff"
PALETTE   = ["#00e5ff", "#a3e635", "#fb923c", "#f87171", "#c084fc", "#fbbf24"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "axes.grid":         True,
    "font.family":       "monospace",
    "axes.titlecolor":   "#ffffff",
    "axes.titlesize":    14
})

# ── Data Loading Logic ────────────────────────────────────────────────────────
def load_data():
    if not DB_PATH.exists():
        print(f"[WARN] DB not found. Generating synthetic data for visual testing.")
        return _synthetic()

    conn = sqlite3.connect(DB_PATH)
    # Optimized query joining games, reviews, and owners data
    q = """
        SELECT g.appid, g.title, g.developer, g.release_date,
               g.launch_price_cents, g.current_price_cents,
               r.review_score, r.total_reviews,
               s.owners_min, s.owners_max
        FROM games g
        LEFT JOIN reviews_summary r ON g.appid = r.appid
        LEFT JOIN steamspy s ON g.appid = s.appid
        WHERE g.type='game' AND g.is_free=0 AND g.launch_price_cents > 0
    """
    df = pd.read_sql(q, conn)
    
    # Primary Genre Extraction
    try:
        gdf = pd.read_sql("SELECT appid, genre FROM game_genres", conn)
        pg = gdf.groupby("appid")["genre"].first().reset_index()
        pg.columns = ["appid", "primary_genre"]
        df = df.merge(pg, on="appid", how="left")
    except:
        df["primary_genre"] = "Other"
    
    conn.close()
    return _process(df)

def _process(df):
    df["initial_price_php"] = df["launch_price_cents"] / 100.0
    df["current_price_php"] = df["current_price_cents"] / 100.0
    df["price_retention"]   = (df["current_price_php"] / df["initial_price_php"]).clip(0, 1)
    df["discount_depth"]    = 1 - df["price_retention"]
    df["release_date"]      = pd.to_datetime(df["release_date"], errors="coerce")
    df["days_since_release"] = (pd.Timestamp.today() - df["release_date"]).dt.days.astype(float)
    df["owners"]            = (df["owners_min"].fillna(0) + df["owners_max"].fillna(0)) / 2
    df["log_ownership"]     = np.log10(df["owners"].replace(0, 1))
    df["review_score"]      = df["review_score"].fillna(50)
    
    # Tiering logic
    AAA = ["valve", "ea", "ubisoft", "activision", "2k", "bethesda", "rockstar", "square enix", "capcom", "sega"]
    df["developer_tier"] = df["developer"].apply(lambda d: "AAA" if any(a in str(d).lower() for a in AAA) else "Mid/Indie")
    
    def _retention_tier(r):
        if r > 0.85: return "Premium Hold"
        if r > 0.50: return "Standard"
        return "Bargain"
    df["value_retention_tier"] = df["price_retention"].apply(_retention_tier)
    
    return df.dropna(subset=["days_since_release"])

def _synthetic():
    # Helper for generating test visuals without the DB
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "initial_price_php": rng.choice([500, 1500, 2500], n),
        "days_since_release": rng.uniform(30, 3000, n),
        "review_score": rng.uniform(40, 95, n),
        "log_ownership": rng.uniform(3, 7, n),
        "primary_genre": rng.choice(["Action", "Indie", "RPG", "Strategy"], n),
        "developer_tier": rng.choice(["AAA", "Mid/Indie"], n)
    })
    df["price_retention"] = np.clip(np.exp(-df["days_since_release"]/2000) + rng.normal(0, 0.1, n), 0.1, 1.0)
    df["current_price_php"] = df["initial_price_php"] * df["price_retention"]
    df["value_retention_tier"] = df["price_retention"].apply(lambda r: "Premium Hold" if r > 0.8 else "Standard")
    df["discount_depth"] = 1 - df["price_retention"]
    return df

# ── Visual Generation Functions ──────────────────────────────────────────────
def savefig(name):
    plt.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  [SAVED] {name}")

def plot_depreciation_curves(df):
    print("Generating: Price Depreciation Curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_genres = df["primary_genre"].value_counts().head(5).index
    
    for i, genre in enumerate(top_genres):
        gdf = df[df["primary_genre"] == genre].copy()
        sns.regplot(data=gdf, x="days_since_release", y=gdf["price_retention"]*100, 
                    scatter=False, label=genre, color=PALETTE[i], ax=ax, lowess=True)
                    
    ax.set_ylim(0, 110)
    ax.set_title("Price Retention Decay by Genre")
    ax.set_xlabel("Days Since Release")
    ax.set_ylabel("Retention (%)")
    ax.legend()
    savefig("price_depreciation_curves.png")

def plot_feature_importance(df):
    print("Generating: ML Feature Importance...")
    le = LabelEncoder()
    dmod = df.copy()
    dmod["developer_tier"] = le.fit_transform(dmod["developer_tier"])
    dmod["primary_genre"] = le.fit_transform(dmod["primary_genre"].astype(str))
    
    feats = ["days_since_release", "review_score", "log_ownership", "initial_price_php", "developer_tier"]
    X = dmod[feats]; y = dmod["price_retention"]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    
    plt.figure(figsize=(8, 5))
    importance = pd.Series(rf.feature_importances_, index=feats).sort_values()
    importance.plot(kind='barh', color=ACCENT)
    plt.title("Drivers of Price Retention (Random Forest)")
    savefig("feature_importance.png")

def plot_review_impact(df):
    print("Generating: Review Score vs Retention...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="review_score", y=df["price_retention"]*100, 
                    hue="value_retention_tier", palette="viridis", alpha=0.5, s=20)
    plt.title("Correlation: Quality (Review Score) vs Price Stability")
    plt.xlabel("Review Score (%)")
    plt.ylabel("Price Retention (%)")
    savefig("review_score_vs_retention.png")

# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("-" * 30)
    print("STATIC VISUAL GENERATOR")
    print("-" * 30)
    
    df_clean = load_data()
    print(f"Loaded {len(df_clean)} records.")
    
    plot_depreciation_curves(df_clean)
    plot_feature_importance(df_clean)
    plot_review_impact(df_clean)
    
    print(f"\nSuccess! All visuals stored in: {OUT_DIR.resolve()}")
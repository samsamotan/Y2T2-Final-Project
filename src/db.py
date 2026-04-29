"""SQLite database for the SteamSale project.

One DB, one schema, idempotent. Call `init_db(connect())` once at the start
of any pipeline run; the schema uses `IF NOT EXISTS` so it's safe to re-run.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "steam.db"


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- The pool of appids we plan to (or have) collect(ed). Acts as a worklist
-- with per-stage progress flags so the pipeline is resumable.
CREATE TABLE IF NOT EXISTS app_list (
    appid              INTEGER PRIMARY KEY,
    name               TEXT,
    source             TEXT,                 -- 'steamspy_top', 'steam_full', etc.
    added_at           TEXT NOT NULL,
    has_details        INTEGER DEFAULT 0,
    has_reviews        INTEGER DEFAULT 0,
    has_steamspy       INTEGER DEFAULT 0,
    has_itad_id        INTEGER DEFAULT 0,
    has_price_history  INTEGER DEFAULT 0,
    has_steamcharts    INTEGER DEFAULT 0,
    last_error         TEXT
);

-- Steam Storefront /api/appdetails (the main game record).
-- Prices are stored in the smallest currency unit (centavos for PHP) as
-- Steam returns them, to avoid float drift.
CREATE TABLE IF NOT EXISTS games (
    appid                INTEGER PRIMARY KEY,
    title                TEXT,
    type                 TEXT,
    is_free              INTEGER,
    short_description    TEXT,
    developer            TEXT,
    publisher            TEXT,
    release_date         TEXT,
    coming_soon          INTEGER,

    currency             TEXT,
    launch_price_cents   INTEGER,            -- Steam "initial"
    current_price_cents  INTEGER,            -- Steam "final"
    discount_percent     INTEGER,

    metacritic_score     INTEGER,
    metacritic_url       TEXT,

    windows              INTEGER,
    mac                  INTEGER,
    linux                INTEGER,

    controller_support   TEXT,
    achievements_total   INTEGER,
    required_age         INTEGER,
    header_image         TEXT,

    fetched_at           TEXT NOT NULL,
    raw_json             TEXT,
    FOREIGN KEY (appid) REFERENCES app_list(appid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS game_genres (
    appid INTEGER NOT NULL,
    genre TEXT NOT NULL,
    PRIMARY KEY (appid, genre),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS game_categories (
    appid    INTEGER NOT NULL,
    category TEXT NOT NULL,
    PRIMARY KEY (appid, category),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

-- Aggregated review stats from /appreviews.
CREATE TABLE IF NOT EXISTS reviews_summary (
    appid              INTEGER PRIMARY KEY,
    total_reviews      INTEGER,
    total_positive     INTEGER,
    total_negative     INTEGER,
    review_score       INTEGER,
    review_score_desc  TEXT,
    fetched_at         TEXT NOT NULL,
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

-- Individual review rows (timestamps drive review-velocity in Part 2).
CREATE TABLE IF NOT EXISTS review_timestamps (
    review_id          INTEGER PRIMARY KEY,
    appid              INTEGER NOT NULL,
    timestamp_created  INTEGER NOT NULL,     -- unix seconds
    voted_up           INTEGER,
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_reviews_appid    ON review_timestamps(appid);
CREATE INDEX IF NOT EXISTS ix_reviews_created  ON review_timestamps(timestamp_created);

-- SteamSpy aggregate stats.
CREATE TABLE IF NOT EXISTS steamspy (
    appid             INTEGER PRIMARY KEY,
    owners_min        INTEGER,
    owners_max        INTEGER,
    average_forever   INTEGER,
    average_2weeks    INTEGER,
    median_forever    INTEGER,
    median_2weeks     INTEGER,
    ccu               INTEGER,
    score_rank        TEXT,
    fetched_at        TEXT NOT NULL,
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steamspy_tags (
    appid INTEGER NOT NULL,
    tag   TEXT NOT NULL,
    votes INTEGER,
    PRIMARY KEY (appid, tag),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

-- IsThereAnyDeal id mapping (their UUID per game) so we can hit history/v2.
CREATE TABLE IF NOT EXISTS itad_mapping (
    appid       INTEGER PRIMARY KEY,
    itad_id     TEXT NOT NULL,
    slug        TEXT,
    fetched_at  TEXT NOT NULL,
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);

-- Historical price points from ITAD. Each row is one snapshot for one shop.
-- Sale events are derived in analysis by grouping consecutive cut>0 rows.
CREATE TABLE IF NOT EXISTS price_history (
    appid            INTEGER NOT NULL,
    shop_id          INTEGER NOT NULL,
    shop_name        TEXT,
    timestamp        TEXT NOT NULL,         -- ISO 8601 UTC
    price_amount     REAL,
    price_currency   TEXT,
    regular_amount   REAL,
    cut              INTEGER,                -- discount percent
    deal_id          TEXT,
    PRIMARY KEY (appid, shop_id, timestamp),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_price_history_appid     ON price_history(appid);
CREATE INDEX IF NOT EXISTS ix_price_history_timestamp ON price_history(timestamp);

-- Concurrent player snapshots from Steam Web API. Single-shot endpoint, so
-- to build a time series the collector must run repeatedly (e.g. daily cron).
CREATE TABLE IF NOT EXISTS player_counts (
    appid         INTEGER NOT NULL,
    fetched_at    TEXT NOT NULL,
    player_count  INTEGER,
    PRIMARY KEY (appid, fetched_at),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_player_counts_appid ON player_counts(appid);

-- Historical CCU scraped from steamcharts.com/app/{appid}/chart-data.json.
-- One row per timestamped data point (typically hourly or every few hours,
-- going back ~4 years). This is the primary input for sale-uplift analysis.
CREATE TABLE IF NOT EXISTS steamcharts_history (
    appid         INTEGER NOT NULL,
    timestamp     TEXT NOT NULL,           -- ISO 8601 UTC
    player_count  REAL,                    -- can be fractional (averaged)
    PRIMARY KEY (appid, timestamp),
    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_steamcharts_appid     ON steamcharts_history(appid);
CREATE INDEX IF NOT EXISTS ix_steamcharts_timestamp ON steamcharts_history(timestamp);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def add_to_app_list(
    conn: sqlite3.Connection,
    apps: list[tuple[int, str]],
    source: str = "manual",
) -> int:
    """Insert (appid, name) pairs into app_list. Existing rows are left alone.
    Returns the number of newly inserted rows.
    """
    now = utcnow_iso()
    rows = [(appid, name, source, now) for appid, name in apps]
    cur = conn.executemany(
        """INSERT OR IGNORE INTO app_list (appid, name, source, added_at)
           VALUES (?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


def mark_progress(
    conn: sqlite3.Connection,
    appid: int,
    flag: str,
    value: int = 1,
    error: str | None = None,
) -> None:
    """Set a progress flag for one appid. `flag` must be one of the has_*
    columns on app_list."""
    allowed = {
        "has_details", "has_reviews", "has_steamspy",
        "has_itad_id", "has_price_history", "has_steamcharts",
    }
    if flag not in allowed:
        raise ValueError(f"unknown flag: {flag}")
    conn.execute(
        f"UPDATE app_list SET {flag} = ?, last_error = ? WHERE appid = ?",
        (value, error, appid),
    )
    conn.commit()


def pending_appids(conn: sqlite3.Connection, flag: str, limit: int | None = None) -> list[int]:
    """Return appids where the given progress flag is 0."""
    sql = f"SELECT appid FROM app_list WHERE {flag} = 0 ORDER BY appid"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql)]

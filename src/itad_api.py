"""IsThereAnyDeal v2 API fetchers.

Two-step flow per game:
  1. Translate Steam appid -> ITAD UUID via /games/lookup/v2
  2. Pull historical price points via /games/history/v2 (Steam shop = 61)

History-window note (matters):
  ITAD's /games/history/v2 caps the response at the last 3 months by default,
  unless `since` is provided. Even with `since`, regional coverage varies
  dramatically — country='PH' returns ~3 events per game, country='US' returns
  20–100+ events going back to ~2012-2015.

  We default to country='US' for the price history collection because the
  `cut` (discount percent) field is currency-agnostic — it's the same number
  whether the price is in USD or PHP. Absolute-price columns (price_amount,
  regular_amount) are then in USD; downstream analyses should rely on `cut`
  rather than mixing currencies. Steam Storefront is still queried with
  cc=ph for the snapshot prices in the games table.
"""
from __future__ import annotations

import sqlite3

from tqdm.auto import tqdm

from .db import mark_progress, utcnow_iso
from .utils import Throttle, get_with_retry

BASE = "https://api.isthereanydeal.com"
STEAM_SHOP_ID = 61

# 2010-01-01 is well before any modern Steam title; ITAD will return whatever
# data exists from that point forward (typically going back to 2012-2015).
DEFAULT_SINCE = "2010-01-01T00:00:00Z"

_throttle = Throttle(min_interval=0.6)


# --------------------------------------------------------------------------
# Step 1 — appid -> ITAD UUID
# --------------------------------------------------------------------------
def lookup_appid(api_key: str, appid: int) -> dict | None:
    """GET /games/lookup/v1?key=&appid= -> {found, game: {id, slug, title}}"""
    r = get_with_retry(
        f"{BASE}/games/lookup/v1",
        params={"key": api_key, "appid": appid},
        throttle=_throttle,
    )
    if r is None:
        return None
    payload = r.json() or {}
    if not payload.get("found"):
        return None
    return payload.get("game")


def store_itad_mapping(conn: sqlite3.Connection, appid: int, game: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO itad_mapping (appid, itad_id, slug, fetched_at)
           VALUES (?, ?, ?, ?)""",
        (appid, game.get("id"), game.get("slug"), utcnow_iso()),
    )
    conn.commit()


def collect_itad_mappings(
    conn: sqlite3.Connection,
    api_key: str,
    appids: list[int],
) -> dict[str, int]:
    stats = {"ok": 0, "missing": 0, "error": 0}
    for appid in tqdm(appids, desc="ITAD lookup", unit="game"):
        try:
            game = lookup_appid(api_key, appid)
        except Exception as e:
            mark_progress(conn, appid, "has_itad_id", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if game is None or not game.get("id"):
            mark_progress(conn, appid, "has_itad_id", value=0, error="not_found")
            stats["missing"] += 1
            continue

        try:
            store_itad_mapping(conn, appid, game)
        except Exception as e:
            conn.rollback()
            mark_progress(conn, appid, "has_itad_id", value=0, error=f"store: {e}"[:200])
            stats["error"] += 1
            continue

        mark_progress(conn, appid, "has_itad_id", value=1)
        stats["ok"] += 1
    return stats


# --------------------------------------------------------------------------
# Step 2 — price history
# --------------------------------------------------------------------------
def fetch_price_history(
    api_key: str,
    itad_id: str,
    *,
    shops: list[int] | None = None,
    country: str = "US",
    since: str | None = DEFAULT_SINCE,
) -> list[dict]:
    """GET /games/history/v2 — list of {timestamp, shop, deal: {price, regular, cut}}.

    Parameters
    ----------
    country :
        ISO 3166-1 alpha-2. Defaults to 'US' for richest historical coverage.
        'PH' has very thin history (~3 months / ~3 events per game). 'US' returns
        20–100+ events going back to 2012-2015 for typical Steam games.
    since :
        ISO 8601 datetime string. ITAD defaults to "last 3 months" if absent;
        we override to fetch the maximum historical window available. Set to
        None to use ITAD's 3-month default.
    """
    params = {
        "key": api_key,
        "id": itad_id,
        "country": country,
    }
    if shops:
        params["shops"] = ",".join(str(s) for s in shops)
    if since:
        params["since"] = since
    r = get_with_retry(f"{BASE}/games/history/v2", params=params, throttle=_throttle)
    if r is None:
        return []
    return r.json() or []


def store_price_history(
    conn: sqlite3.Connection,
    appid: int,
    history: list[dict],
) -> int:
    """Insert price-history rows (idempotent on (appid, shop_id, timestamp)).

    Each entry from /games/history/v2 looks like:
      {timestamp, shop: {id, name}, deal: {price, regular, cut, ...}}
    `shop` is at the top level, not under `deal`.
    """
    rows = []
    for entry in history:
        shop = entry.get("shop") or {}
        deal = entry.get("deal") or {}
        price = deal.get("price") or {}
        regular = deal.get("regular") or {}
        rows.append(
            (
                appid,
                shop.get("id") or 0,
                shop.get("name"),
                entry.get("timestamp"),
                price.get("amount"),
                price.get("currency"),
                regular.get("amount"),
                deal.get("cut"),
                deal.get("uuid") or deal.get("id"),
            )
        )
    if not rows:
        return 0
    cur = conn.executemany(
        """INSERT OR IGNORE INTO price_history (
            appid, shop_id, shop_name, timestamp,
            price_amount, price_currency, regular_amount,
            cut, deal_id
        ) VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount or 0


def reset_price_history(conn: sqlite3.Connection) -> dict[str, int]:
    """Drop all price_history rows and reset has_price_history flags to 0.

    Use this before re-running collect_price_history if you want a fresh full
    refresh — e.g., after switching country / since to fetch deeper history.
    Idempotent: safe to call when there's already nothing to clear.

    Uses ``DROP TABLE`` + recreate rather than ``DELETE FROM`` — much faster
    on a half-million-row table (milliseconds instead of 30-60s) because
    SQLite can free pages wholesale instead of walking every row + updating
    indexes one by one. No network calls; this only touches the local DB.
    """
    from .db import init_db

    n_rows = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
    n_flagged = conn.execute("SELECT COUNT(*) FROM app_list WHERE has_price_history = 1").fetchone()[0]

    # Drop + recreate the table (fast). init_db's CREATE TABLE IF NOT EXISTS
    # rebuilds the schema from src/db.py — single source of truth.
    conn.execute("DROP TABLE IF EXISTS price_history")
    init_db(conn)

    conn.execute("UPDATE app_list SET has_price_history = 0, last_error = NULL WHERE has_price_history = 1")
    conn.commit()
    print(f"reset_price_history: cleared {n_rows:,} rows; reset {n_flagged:,} has_price_history flags")
    return {"rows_cleared": n_rows, "flags_reset": n_flagged}


def collect_price_history(
    conn: sqlite3.Connection,
    api_key: str,
    *,
    shops: list[int] | None = None,
    country: str = "US",
    since: str | None = DEFAULT_SINCE,
) -> dict[str, int]:
    """Pull history for every appid that has an itad_id but no price_history yet.

    Defaults to country='US' and since=DEFAULT_SINCE for richest historical
    coverage. Call ``reset_price_history(conn)`` first if you want to refresh
    games already flagged has_price_history=1 (e.g., after changing country
    or since).
    """
    if shops is None:
        shops = [STEAM_SHOP_ID]

    rows = conn.execute(
        """SELECT a.appid, m.itad_id
           FROM app_list a
           JOIN itad_mapping m ON m.appid = a.appid
           WHERE a.has_itad_id = 1 AND a.has_price_history = 0"""
    ).fetchall()

    stats = {"ok": 0, "empty": 0, "error": 0, "rows_inserted": 0}
    for appid, itad_id in tqdm(rows, desc="ITAD price history", unit="game"):
        try:
            history = fetch_price_history(
                api_key, itad_id, shops=shops, country=country, since=since,
            )
        except Exception as e:
            mark_progress(conn, appid, "has_price_history", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if not history:
            mark_progress(conn, appid, "has_price_history", value=1, error="empty")
            stats["empty"] += 1
            continue

        try:
            inserted = store_price_history(conn, appid, history)
        except Exception as e:
            conn.rollback()
            mark_progress(conn, appid, "has_price_history", value=0, error=f"store: {e}"[:200])
            stats["error"] += 1
            continue

        stats["rows_inserted"] += inserted
        mark_progress(conn, appid, "has_price_history", value=1)
        stats["ok"] += 1
    return stats

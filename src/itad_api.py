"""IsThereAnyDeal v2 API fetchers.

Two-step flow per game:
  1. Translate Steam appid -> ITAD UUID via /games/lookup/v2
  2. Pull historical price points via /games/history/v2 (Steam shop = 61)
"""
from __future__ import annotations

import sqlite3

from .db import mark_progress, utcnow_iso
from .utils import Throttle, get_with_retry

BASE = "https://api.isthereanydeal.com"
STEAM_SHOP_ID = 61

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
    for appid in appids:
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

        store_itad_mapping(conn, appid, game)
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
    country: str = "PH",
) -> list[dict]:
    """GET /games/history/v2 — list of {timestamp, deal: {price, regular, cut, shop}}"""
    params = {
        "key": api_key,
        "id": itad_id,
        "country": country,
    }
    if shops:
        params["shops"] = ",".join(str(s) for s in shops)
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


def collect_price_history(
    conn: sqlite3.Connection,
    api_key: str,
    *,
    shops: list[int] | None = None,
    country: str = "PH",
) -> dict[str, int]:
    """Pull history for every appid that has an itad_id but no price_history yet."""
    if shops is None:
        shops = [STEAM_SHOP_ID]

    rows = conn.execute(
        """SELECT a.appid, m.itad_id
           FROM app_list a
           JOIN itad_mapping m ON m.appid = a.appid
           WHERE a.has_itad_id = 1 AND a.has_price_history = 0"""
    ).fetchall()

    stats = {"ok": 0, "empty": 0, "error": 0, "rows_inserted": 0}
    for appid, itad_id in rows:
        try:
            history = fetch_price_history(api_key, itad_id, shops=shops, country=country)
        except Exception as e:
            mark_progress(conn, appid, "has_price_history", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if not history:
            mark_progress(conn, appid, "has_price_history", value=1, error="empty")
            stats["empty"] += 1
            continue

        inserted = store_price_history(conn, appid, history)
        stats["rows_inserted"] += inserted
        mark_progress(conn, appid, "has_price_history", value=1)
        stats["ok"] += 1
    return stats

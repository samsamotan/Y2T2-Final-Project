"""Steam Storefront and Web API fetchers.

Storefront is undocumented and unauthenticated; the Web API uses STEAM_API_KEY.
Storefront is the bottleneck — we throttle it conservatively to ~1 req / 1.5 s.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import mark_progress, utcnow_iso
from .utils import Throttle, get_with_retry

STOREFRONT = "https://store.steampowered.com/api"
STORE_BASE = "https://store.steampowered.com"   # /appreviews lives here, not under /api
WEB_API = "https://api.steampowered.com"

_storefront_throttle = Throttle(min_interval=1.5)
_webapi_throttle = Throttle(min_interval=0.5)


# --------------------------------------------------------------------------
# Master app list (every Steam app — used as a fallback sampling pool).
# --------------------------------------------------------------------------
def fetch_full_app_list() -> list[dict[str, Any]]:
    """ISteamApps/GetAppList — returns every app (~200k+) with appid + name.
    Includes DLC, demos, software; filter downstream."""
    url = f"{WEB_API}/ISteamApps/GetAppList/v2/"
    r = get_with_retry(url, throttle=_webapi_throttle, timeout=60)
    if r is None:
        return []
    return r.json().get("applist", {}).get("apps", [])


# --------------------------------------------------------------------------
# Storefront /api/appdetails
# --------------------------------------------------------------------------
def fetch_app_details(appid: int, cc: str = "ph", lang: str = "english") -> dict | None:
    """Hit /api/appdetails for one appid. Returns the inner `data` dict
    on success, or None if Steam said success=false."""
    url = f"{STOREFRONT}/appdetails"
    params = {"appids": str(appid), "cc": cc, "l": lang}
    r = get_with_retry(url, params=params, throttle=_storefront_throttle)
    if r is None:
        return None
    payload = r.json() or {}
    entry = payload.get(str(appid))
    if not entry or not entry.get("success"):
        return None
    return entry.get("data")


def store_app_details(conn: sqlite3.Connection, appid: int, data: dict) -> None:
    """Decompose one /appdetails response into games, game_genres,
    game_categories. Idempotent — re-running overwrites the games row and
    replaces the child rows."""
    price = data.get("price_overview") or {}
    platforms = data.get("platforms") or {}
    metacritic = data.get("metacritic") or {}
    release = data.get("release_date") or {}
    achievements = (data.get("achievements") or {}).get("total")

    row = (
        appid,
        data.get("name"),
        data.get("type"),
        1 if data.get("is_free") else 0,
        data.get("short_description"),
        ", ".join(data.get("developers") or []) or None,
        ", ".join(data.get("publishers") or []) or None,
        release.get("date"),
        1 if release.get("coming_soon") else 0,
        price.get("currency"),
        price.get("initial"),
        price.get("final"),
        price.get("discount_percent"),
        metacritic.get("score"),
        metacritic.get("url"),
        1 if platforms.get("windows") else 0,
        1 if platforms.get("mac") else 0,
        1 if platforms.get("linux") else 0,
        data.get("controller_support"),
        achievements,
        data.get("required_age"),
        data.get("header_image"),
        utcnow_iso(),
        json.dumps(data, ensure_ascii=False),
    )

    conn.execute(
        """INSERT OR REPLACE INTO games (
            appid, title, type, is_free, short_description,
            developer, publisher, release_date, coming_soon,
            currency, launch_price_cents, current_price_cents, discount_percent,
            metacritic_score, metacritic_url,
            windows, mac, linux,
            controller_support, achievements_total, required_age, header_image,
            fetched_at, raw_json
        ) VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?, ?,?,?,?, ?,?)""",
        row,
    )

    # Dedupe via set — Steam occasionally returns the same genre/category twice
    # in the same response, which otherwise trips the (appid, x) PK.
    conn.execute("DELETE FROM game_genres WHERE appid = ?", (appid,))
    genre_set = {g["description"] for g in (data.get("genres") or []) if g.get("description")}
    if genre_set:
        conn.executemany(
            "INSERT INTO game_genres (appid, genre) VALUES (?, ?)",
            [(appid, g) for g in genre_set],
        )

    conn.execute("DELETE FROM game_categories WHERE appid = ?", (appid,))
    cat_set = {c["description"] for c in (data.get("categories") or []) if c.get("description")}
    if cat_set:
        conn.executemany(
            "INSERT INTO game_categories (appid, category) VALUES (?, ?)",
            [(appid, c) for c in cat_set],
        )

    conn.commit()


def collect_app_details(
    conn: sqlite3.Connection,
    appids: list[int],
    *,
    cc: str = "ph",
    skip_non_games: bool = True,
) -> dict[str, int]:
    """Fetch + store details for each appid; updates app_list.has_details.
    Returns a tally dict."""
    stats = {"ok": 0, "missing": 0, "skipped": 0, "error": 0}
    for appid in appids:
        try:
            data = fetch_app_details(appid, cc=cc)
        except Exception as e:
            mark_progress(conn, appid, "has_details", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if data is None:
            mark_progress(conn, appid, "has_details", value=0, error="not_found")
            stats["missing"] += 1
            continue

        if skip_non_games and data.get("type") != "game":
            mark_progress(conn, appid, "has_details", value=1, error=f"type={data.get('type')}")
            stats["skipped"] += 1
            continue

        try:
            store_app_details(conn, appid, data)
        except Exception as e:
            # One bad row shouldn't kill the whole batch — roll back the partial
            # transaction, log the error, and move on.
            conn.rollback()
            mark_progress(conn, appid, "has_details", value=0, error=f"store: {e}"[:200])
            stats["error"] += 1
            continue

        mark_progress(conn, appid, "has_details", value=1)
        stats["ok"] += 1
    return stats


# --------------------------------------------------------------------------
# /appreviews — summary + paginated detail
# --------------------------------------------------------------------------
def fetch_review_summary(appid: int) -> dict | None:
    url = f"{STORE_BASE}/appreviews/{appid}"
    params = {"json": "1", "language": "all", "purchase_type": "all", "num_per_page": "0"}
    r = get_with_retry(url, params=params, throttle=_storefront_throttle)
    if r is None:
        return None
    payload = r.json() or {}
    if not payload.get("success"):
        return None
    return payload.get("query_summary")


def store_review_summary(conn: sqlite3.Connection, appid: int, summary: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO reviews_summary (
            appid, total_reviews, total_positive, total_negative,
            review_score, review_score_desc, fetched_at
        ) VALUES (?,?,?,?,?,?,?)""",
        (
            appid,
            summary.get("total_reviews"),
            summary.get("total_positive"),
            summary.get("total_negative"),
            summary.get("review_score"),
            summary.get("review_score_desc"),
            utcnow_iso(),
        ),
    )
    conn.commit()


def fetch_review_timestamps(appid: int, max_pages: int = 10) -> list[dict]:
    """Page through /appreviews collecting (review_id, timestamp_created, voted_up).
    `max_pages` * 100 caps how deep we go per game — review velocity only needs
    recent ones, so this is usually plenty."""
    url = f"{STORE_BASE}/appreviews/{appid}"
    cursor = "*"
    out: list[dict] = []
    for _ in range(max_pages):
        params = {
            "json": "1",
            "language": "all",
            "purchase_type": "all",
            "num_per_page": "100",
            "filter": "recent",
            "cursor": cursor,
        }
        r = get_with_retry(url, params=params, throttle=_storefront_throttle)
        if r is None:
            break
        payload = r.json() or {}
        if not payload.get("success"):
            break
        reviews = payload.get("reviews") or []
        if not reviews:
            break
        for rv in reviews:
            out.append(
                {
                    "review_id": int(rv["recommendationid"]),
                    "timestamp_created": rv.get("timestamp_created"),
                    "voted_up": 1 if rv.get("voted_up") else 0,
                }
            )
        next_cursor = payload.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return out


def store_review_timestamps(conn: sqlite3.Connection, appid: int, reviews: list[dict]) -> None:
    if not reviews:
        return
    conn.executemany(
        """INSERT OR IGNORE INTO review_timestamps
           (review_id, appid, timestamp_created, voted_up)
           VALUES (?, ?, ?, ?)""",
        [(r["review_id"], appid, r["timestamp_created"], r["voted_up"]) for r in reviews],
    )
    conn.commit()


def collect_reviews(
    conn: sqlite3.Connection,
    appids: list[int],
    *,
    fetch_individual: bool = False,
    max_pages: int = 5,
) -> dict[str, int]:
    stats = {"ok": 0, "missing": 0, "error": 0}
    for appid in appids:
        try:
            summary = fetch_review_summary(appid)
        except Exception as e:
            mark_progress(conn, appid, "has_reviews", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if summary is None:
            mark_progress(conn, appid, "has_reviews", value=0, error="not_found")
            stats["missing"] += 1
            continue

        store_review_summary(conn, appid, summary)
        if fetch_individual:
            reviews = fetch_review_timestamps(appid, max_pages=max_pages)
            store_review_timestamps(conn, appid, reviews)

        mark_progress(conn, appid, "has_reviews", value=1)
        stats["ok"] += 1
    return stats


# --------------------------------------------------------------------------
# Concurrent player count (single-shot — collect repeatedly to build a series)
# --------------------------------------------------------------------------
def fetch_current_players(appid: int) -> int | None:
    url = f"{WEB_API}/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
    r = get_with_retry(url, params={"appid": appid}, throttle=_webapi_throttle)
    if r is None:
        return None
    resp = (r.json() or {}).get("response") or {}
    if resp.get("result") != 1:
        return None
    return resp.get("player_count")


def store_player_count(conn: sqlite3.Connection, appid: int, count: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO player_counts (appid, fetched_at, player_count) VALUES (?, ?, ?)",
        (appid, utcnow_iso(), count),
    )
    conn.commit()


def collect_player_counts(conn: sqlite3.Connection, appids: list[int]) -> dict[str, int]:
    stats = {"ok": 0, "missing": 0}
    for appid in appids:
        count = fetch_current_players(appid)
        if count is None:
            stats["missing"] += 1
            continue
        store_player_count(conn, appid, count)
        stats["ok"] += 1
    return stats

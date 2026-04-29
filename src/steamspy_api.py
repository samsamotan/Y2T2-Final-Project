"""SteamSpy API fetchers.

SteamSpy throttles the per-app `appdetails` request to ~1/sec, but the bulk
`all` and `top100*` requests are limited to 1/min. We honor both with a
shared throttle for per-app calls and a separate one for bulk pulls.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from .db import mark_progress, utcnow_iso
from .utils import Throttle, get_with_retry

BASE = "https://steamspy.com/api.php"

_per_app_throttle = Throttle(min_interval=1.1)
_bulk_throttle = Throttle(min_interval=61.0)


# --------------------------------------------------------------------------
# Bulk listing — used for sampling a top-N pool stratified by ownership.
# --------------------------------------------------------------------------
def fetch_top_owned_page(page: int) -> dict[str, dict[str, Any]]:
    """`request=all&page=N` — page N of top-owned games (1000 per page).
    page=0..3 covers the top 4000; pages beyond that exist but coverage thins.
    """
    r = get_with_retry(
        BASE,
        params={"request": "all", "page": page},
        throttle=_bulk_throttle,
        timeout=60,
    )
    if r is None:
        return {}
    return r.json() or {}


def fetch_sample_pool(num_pages: int = 5) -> list[dict[str, Any]]:
    """Pull `num_pages` pages of top-owned games (1000 each)."""
    out: list[dict[str, Any]] = []
    for p in range(num_pages):
        page = fetch_top_owned_page(p)
        for entry in page.values():
            if "appid" in entry and "name" in entry:
                out.append({"appid": int(entry["appid"]), "name": entry["name"]})
    return out


# --------------------------------------------------------------------------
# Per-app details
# --------------------------------------------------------------------------
def fetch_appdetails(appid: int) -> dict | None:
    r = get_with_retry(
        BASE,
        params={"request": "appdetails", "appid": appid},
        throttle=_per_app_throttle,
    )
    if r is None:
        return None
    data = r.json() or {}
    # SteamSpy returns {"appid": null, ...} for unknown ids.
    if not data.get("appid"):
        return None
    return data


def _parse_owners(owners: str | None) -> tuple[int | None, int | None]:
    """`owners` looks like '1,000,000 .. 2,000,000' — split into (min, max)."""
    if not owners:
        return (None, None)
    try:
        lo, hi = owners.split("..")
        return (int(lo.replace(",", "").strip()), int(hi.replace(",", "").strip()))
    except (ValueError, AttributeError):
        return (None, None)


def store_appdetails(conn: sqlite3.Connection, appid: int, data: dict) -> None:
    owners_min, owners_max = _parse_owners(data.get("owners"))
    conn.execute(
        """INSERT OR REPLACE INTO steamspy (
            appid, owners_min, owners_max,
            average_forever, average_2weeks, median_forever, median_2weeks,
            ccu, score_rank, fetched_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            appid,
            owners_min,
            owners_max,
            data.get("average_forever"),
            data.get("average_2weeks"),
            data.get("median_forever"),
            data.get("median_2weeks"),
            data.get("ccu"),
            str(data.get("score_rank")) if data.get("score_rank") not in (None, "") else None,
            utcnow_iso(),
        ),
    )

    conn.execute("DELETE FROM steamspy_tags WHERE appid = ?", (appid,))
    tags = data.get("tags") or {}
    if isinstance(tags, dict) and tags:
        conn.executemany(
            "INSERT INTO steamspy_tags (appid, tag, votes) VALUES (?, ?, ?)",
            [(appid, t, int(v)) for t, v in tags.items()],
        )

    conn.commit()


def collect_steamspy(conn: sqlite3.Connection, appids: list[int]) -> dict[str, int]:
    stats = {"ok": 0, "missing": 0, "error": 0}
    for appid in appids:
        try:
            data = fetch_appdetails(appid)
        except Exception as e:
            mark_progress(conn, appid, "has_steamspy", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if data is None:
            mark_progress(conn, appid, "has_steamspy", value=0, error="not_found")
            stats["missing"] += 1
            continue

        store_appdetails(conn, appid, data)
        mark_progress(conn, appid, "has_steamspy", value=1)
        stats["ok"] += 1
    return stats

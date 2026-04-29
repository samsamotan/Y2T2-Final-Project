"""SteamCharts historical-CCU scraper.

SteamCharts has an undocumented JSON endpoint at
`steamcharts.com/app/{appid}/chart-data.json` that returns the same data
its on-page chart consumes — typically an array of [unix_ms, player_count]
pairs going back ~4 years.

We throttle to 2s/request and send a descriptive User-Agent (don't be a
silent bot — they're a small free site doing us a favor).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .db import mark_progress
from .utils import Throttle, get_with_retry

BASE = "https://steamcharts.com"
HEADERS = {
    "User-Agent": (
        "SteamSale-Research/1.0 (academic project; "
        "github.com/samsamotan/Y2T2-Final-Project)"
    ),
}

_throttle = Throttle(min_interval=2.0)


def fetch_history(appid: int) -> list[tuple[int, float]]:
    """GET /app/{appid}/chart-data.json — list of [unix_ms, player_count].
    Returns [] if the page 404s or the body isn't a JSON list."""
    url = f"{BASE}/app/{appid}/chart-data.json"
    r = get_with_retry(url, headers=HEADERS, throttle=_throttle)
    if r is None:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [(int(ts), float(c)) for ts, c in data if c is not None]


def store_history(
    conn: sqlite3.Connection,
    appid: int,
    points: list[tuple[int, float]],
) -> int:
    if not points:
        return 0
    rows = [
        (
            appid,
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(timespec="seconds"),
            count,
        )
        for ts, count in points
    ]
    cur = conn.executemany(
        """INSERT OR IGNORE INTO steamcharts_history (appid, timestamp, player_count)
           VALUES (?, ?, ?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount or 0


def collect_history(conn: sqlite3.Connection, appids: list[int]) -> dict[str, int]:
    """Fetch + store historical CCU for each appid; updates has_steamcharts."""
    stats = {"ok": 0, "missing": 0, "error": 0, "rows_inserted": 0}
    for appid in appids:
        try:
            points = fetch_history(appid)
        except Exception as e:
            mark_progress(conn, appid, "has_steamcharts", value=0, error=str(e)[:200])
            stats["error"] += 1
            continue

        if not points:
            mark_progress(conn, appid, "has_steamcharts", value=0, error="not_found")
            stats["missing"] += 1
            continue

        inserted = store_history(conn, appid, points)
        stats["rows_inserted"] += inserted
        mark_progress(conn, appid, "has_steamcharts", value=1)
        stats["ok"] += 1
    return stats

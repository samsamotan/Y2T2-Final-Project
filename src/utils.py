"""Shared HTTP helpers: throttling and retry-with-backoff."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


def _load_project_env() -> Path | None:
    """Load the first `.env` found in common project/notebook locations."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            load_dotenv(resolved, override=False)
            return resolved
    return None


def load_keys() -> dict[str, str | None]:
    """Load .env from project root and return the keys we use."""
    env_path = _load_project_env()
    return {
        "steam": os.getenv("STEAM_API_KEY"),
        "itad": os.getenv("ISTHEREANYDEAL_API"),
        "env_path": str(env_path) if env_path else None,
    }


class Throttle:
    """Simple sleep-based throttle: at most one call every `min_interval` s."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
    max_attempts: int = 4,
    backoff: float = 2.0,
    throttle: Throttle | None = None,
) -> requests.Response | None:
    """GET with exponential backoff. Returns the Response on 2xx, or None
    after exhausting retries. 429 and 5xx trigger a retry; 4xx (other) does not.
    """
    for attempt in range(1, max_attempts + 1):
        if throttle is not None:
            throttle.wait()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt == max_attempts:
                print(f"[get_with_retry] giving up on {url}: {e}")
                return None
            time.sleep(backoff ** attempt)
            continue

        if r.status_code == 200:
            return r
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == max_attempts:
                print(f"[get_with_retry] {r.status_code} on {url}, giving up")
                return None
            time.sleep(backoff ** attempt)
            continue
        # Other 4xx — body usually says why; don't retry.
        print(f"[get_with_retry] {r.status_code} on {url}: {r.text[:200]}")
        return None
    return None


def post_with_retry(
    url: str,
    *,
    json_body: dict | list | None = None,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
    max_attempts: int = 4,
    backoff: float = 2.0,
    throttle: Throttle | None = None,
) -> requests.Response | None:
    for attempt in range(1, max_attempts + 1):
        if throttle is not None:
            throttle.wait()
        try:
            r = requests.post(
                url, json=json_body, params=params, headers=headers, timeout=timeout
            )
        except requests.RequestException as e:
            if attempt == max_attempts:
                print(f"[post_with_retry] giving up on {url}: {e}")
                return None
            time.sleep(backoff ** attempt)
            continue

        if r.status_code in (200, 201):
            return r
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == max_attempts:
                print(f"[post_with_retry] {r.status_code} on {url}, giving up")
                return None
            time.sleep(backoff ** attempt)
            continue
        print(f"[post_with_retry] {r.status_code} on {url}: {r.text[:200]}")
        return None
    return None

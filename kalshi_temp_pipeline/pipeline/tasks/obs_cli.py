"""IEM CLI observation client with simple on-disk caching."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx

DEFAULT_BASE_URL = "https://mesonet.agron.iastate.edu"
DEFAULT_CACHE_DIR = Path("artifacts/obs_cache")


def _cache_path(cache_dir: Path, station: str, year: int) -> Path:
    return cache_dir / f"{station.upper()}_{year}.json"


def fetch_cli_year(
    station: str,
    year: int,
    *,
    base_url: str = DEFAULT_BASE_URL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout_s: float = 20.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch IEM CLI JSON for a station/year, using on-disk cache if available."""

    cache_file = _cache_path(cache_dir, station, year)
    if cache_file.exists():
        return cast(dict[str, Any], json.loads(cache_file.read_text(encoding="utf-8")))

    cache_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        with httpx.Client(timeout=timeout_s) as owned_client:
            response = owned_client.get(
                f"{base_url.rstrip('/')}/json/cli.py",
                params={"station": station, "year": year},
            )
            response.raise_for_status()
            payload = cast(dict[str, Any], response.json())
    else:
        response = client.get(
            f"{base_url.rstrip('/')}/json/cli.py",
            params={"station": station, "year": year},
        )
        response.raise_for_status()
        payload = cast(dict[str, Any], response.json())

    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def extract_daily_tmax(cli_json: dict[str, Any]) -> dict[date, float]:
    """Extract daily Tmax map from IEM CLI payload.

    Expected minimal fields per row: `valid` (YYYY-MM-DD...) and `high`.
    Rows missing or invalid fields are skipped.
    """

    rows = cli_json.get("results", [])
    output: dict[date, float] = {}
    for row in rows:
        valid_raw = row.get("valid")
        high_raw = row.get("high")
        if not isinstance(valid_raw, str):
            continue
        if high_raw is None:
            continue
        try:
            day = date.fromisoformat(valid_raw[:10])
            tmax = float(high_raw)
        except (ValueError, TypeError):
            continue
        output[day] = tmax
    return output


class ObsCliClient:
    """Mockable wrapper class for observation fetches."""

    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        self.base_url = base_url
        self.cache_dir = cache_dir

    def fetch_daily_cli(self, station: str, year: int) -> dict[str, Any]:
        """Compatibility wrapper for existing flow code."""

        return fetch_cli_year(station, year, base_url=self.base_url, cache_dir=self.cache_dir)

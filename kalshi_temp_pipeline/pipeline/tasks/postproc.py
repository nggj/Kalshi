"""WRF postprocessing placeholders."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def extract_station_series() -> tuple[list[datetime], list[float]]:
    """Return deterministic sample 6-hourly Fahrenheit points."""

    tzinfo = ZoneInfo("America/Chicago")
    times = [
        datetime(2026, 7, 10, 0, 0, tzinfo=tzinfo),
        datetime(2026, 7, 10, 6, 0, tzinfo=tzinfo),
        datetime(2026, 7, 10, 12, 0, tzinfo=tzinfo),
        datetime(2026, 7, 10, 18, 0, tzinfo=tzinfo),
        datetime(2026, 7, 11, 0, 0, tzinfo=tzinfo),
        datetime(2026, 7, 11, 6, 0, tzinfo=tzinfo),
    ]
    temps = [79.0, 84.0, 91.5, 88.0, 82.0, 80.0]
    return times, temps

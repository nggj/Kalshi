"""Climate-day window helpers aligned to NWS LST rules."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo


def _is_dst_at_local_noon(target_date: date, tzinfo: ZoneInfo) -> bool:
    noon = datetime.combine(target_date, time(12, 0), tzinfo=tzinfo)
    offset = noon.utcoffset()
    if offset is None:
        return False

    jan_reference = datetime(target_date.year, 1, 15, 12, 0, tzinfo=tzinfo)
    jan_offset = jan_reference.utcoffset()
    if jan_offset is None:
        return False

    return offset > jan_offset


def get_climate_day_window(target_date: date, tz: str) -> tuple[datetime, datetime]:
    """Return [start, end) timezone-aware climate-day window.

    Rule (Local Standard Time definition):
    - non-DST date: [00:00 local, next day 00:00 local)
    - DST date: [01:00 local, next day 01:00 local)
    """

    tzinfo = ZoneInfo(tz)
    start = datetime.combine(target_date, time(0, 0), tzinfo=tzinfo)
    if _is_dst_at_local_noon(target_date, tzinfo):
        start = start + timedelta(hours=1)

    end = start + timedelta(days=1)
    return start, end


def tmax_over_window(
    times: Sequence[datetime],
    temps: Sequence[float],
    window: tuple[datetime, datetime],
) -> float:
    """Compute Tmax using points in [start, end).

    Supports hourly or sub-hourly points.
    """

    if len(times) != len(temps):
        raise ValueError("times and temps lengths must match")
    if not times:
        raise ValueError("times/temps cannot be empty")

    start, end = window
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("window datetimes must be timezone-aware")
    if start >= end:
        raise ValueError("window start must be before end")

    selected = [temp for dt, temp in zip(times, temps, strict=True) if start <= dt < end]
    if not selected:
        raise ValueError("No data points in climate-day window")

    return float(max(selected))

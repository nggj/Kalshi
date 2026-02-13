from datetime import date, datetime
from zoneinfo import ZoneInfo

from kalshi_temp_pipeline.pipeline.tasks.time_windows import (
    get_climate_day_window,
    tmax_over_window,
)


def test_get_climate_day_window_non_dst_new_york() -> None:
    start, end = get_climate_day_window(date(2026, 1, 15), "America/New_York")
    assert start == datetime(2026, 1, 15, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    assert end == datetime(2026, 1, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))


def test_get_climate_day_window_dst_new_york() -> None:
    start, end = get_climate_day_window(date(2026, 7, 15), "America/New_York")
    assert start == datetime(2026, 7, 15, 1, 0, tzinfo=ZoneInfo("America/New_York"))
    assert end == datetime(2026, 7, 16, 1, 0, tzinfo=ZoneInfo("America/New_York"))


def test_tmax_over_window_start_inclusive_end_exclusive() -> None:
    tzinfo = ZoneInfo("America/New_York")
    window = get_climate_day_window(date(2026, 7, 15), "America/New_York")

    times = [
        datetime(2026, 7, 15, 1, 0, tzinfo=tzinfo),   # start boundary: include
        datetime(2026, 7, 15, 12, 0, tzinfo=tzinfo),  # include
        datetime(2026, 7, 16, 1, 0, tzinfo=tzinfo),   # end boundary: exclude
    ]
    temps = [80.0, 90.0, 999.0]

    assert tmax_over_window(times, temps, window) == 90.0

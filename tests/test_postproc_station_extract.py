from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.postproc import (
    extract_station_series,
    extract_station_series_from_npz,
)


def _write_plane_npz(npz_path: Path) -> None:
    lats = np.array([0.0, 1.0], dtype=float)
    lons = np.array([0.0, 1.0], dtype=float)
    times_utc = np.array([0, 3600], dtype=np.int64)

    # Kelvin fields; at station (0.25, 0.75), bilinear gives 273.15 + lat + 2*lon + t
    t2m = np.array(
        [
            [[273.15, 275.15], [274.15, 276.15]],
            [[274.15, 276.15], [275.15, 277.15]],
        ],
        dtype=float,
    )
    np.savez(npz_path, lats=lats, lons=lons, times_utc=times_utc, t2m=t2m)


def test_extract_station_series_from_npz_bilinear_and_units(tmp_path: Path) -> None:
    npz_path = tmp_path / "grid.npz"
    _write_plane_npz(npz_path)

    times, temps_f = extract_station_series_from_npz(
        npz_path=npz_path,
        station_lat=0.25,
        station_lon=0.75,
        tz="America/New_York",
        units="K",
    )

    # Expected Celsius values: 1.75, 2.75 -> Fahrenheit: 35.15, 36.95
    np.testing.assert_allclose(temps_f, [35.15, 36.95], atol=1e-6)

    assert times[0].tzinfo == ZoneInfo("America/New_York")
    assert times[0] == datetime(1969, 12, 31, 19, 0, tzinfo=ZoneInfo("America/New_York"))
    assert times[1] == datetime(1969, 12, 31, 20, 0, tzinfo=ZoneInfo("America/New_York"))


def test_extract_station_series_wrapper_uses_npz_when_provided(tmp_path: Path) -> None:
    npz_path = tmp_path / "grid.npz"
    _write_plane_npz(npz_path)

    times, temps = extract_station_series(
        npz_path=npz_path,
        station_lat=0.25,
        station_lon=0.75,
        tz="UTC",
    )

    assert len(times) == 2
    np.testing.assert_allclose(temps, [35.15, 36.95], atol=1e-6)


def test_extract_station_series_fallback_sample() -> None:
    times, temps = extract_station_series()
    assert len(times) == 6
    assert temps[2] == 91.5

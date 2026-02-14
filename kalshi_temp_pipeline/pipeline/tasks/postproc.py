"""Station extraction from gridded model artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def _to_fahrenheit(values: np.ndarray, units: str) -> np.ndarray:
    u = units.upper()
    if u == "K":
        c = values - 273.15
        return c * 9.0 / 5.0 + 32.0
    if u == "C":
        return values * 9.0 / 5.0 + 32.0
    if u == "F":
        return values
    raise ValueError("units must be one of: K, C, F")


def _parse_times_to_local(times_utc: np.ndarray, tz: str) -> list[datetime]:
    tzinfo = ZoneInfo(tz)
    parsed: list[datetime] = []
    if np.issubdtype(times_utc.dtype, np.integer):
        for ts in times_utc:
            dt = datetime.fromtimestamp(int(ts), tz=ZoneInfo("UTC")).astimezone(tzinfo)
            parsed.append(dt)
        return parsed

    for raw in times_utc:
        s = str(raw)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt_utc = datetime.fromisoformat(s)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=ZoneInfo("UTC"))
        parsed.append(dt_utc.astimezone(tzinfo))
    return parsed


def _bilinear_on_regular_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    field_2d: np.ndarray,
    station_lat: float,
    station_lon: float,
) -> float:
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError("lats/lons must be 1D arrays")
    if not (lats[0] <= station_lat <= lats[-1]) or not (lons[0] <= station_lon <= lons[-1]):
        raise ValueError("station coordinate is outside grid bounds")

    iy_hi = int(np.searchsorted(lats, station_lat, side="right"))
    ix_hi = int(np.searchsorted(lons, station_lon, side="right"))

    iy_hi = min(max(iy_hi, 1), len(lats) - 1)
    ix_hi = min(max(ix_hi, 1), len(lons) - 1)
    iy_lo = iy_hi - 1
    ix_lo = ix_hi - 1

    y0 = float(lats[iy_lo])
    y1 = float(lats[iy_hi])
    x0 = float(lons[ix_lo])
    x1 = float(lons[ix_hi])

    q11 = float(field_2d[iy_lo, ix_lo])
    q12 = float(field_2d[iy_hi, ix_lo])
    q21 = float(field_2d[iy_lo, ix_hi])
    q22 = float(field_2d[iy_hi, ix_hi])

    tx = 0.0 if x1 == x0 else (station_lon - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (station_lat - y0) / (y1 - y0)

    return (
        q11 * (1 - tx) * (1 - ty)
        + q21 * tx * (1 - ty)
        + q12 * (1 - tx) * ty
        + q22 * tx * ty
    )


def extract_station_series_from_npz(
    npz_path: Path,
    station_lat: float,
    station_lon: float,
    tz: str,
    *,
    var_name: str = "t2m",
    units: str = "K",
) -> tuple[list[datetime], list[float]]:
    """Extract station series from simple gridded NPZ via bilinear interpolation."""

    data = np.load(npz_path, allow_pickle=False)
    lats = np.asarray(data["lats"], dtype=float)
    lons = np.asarray(data["lons"], dtype=float)
    times_utc = np.asarray(data["times_utc"])
    var = np.asarray(data[var_name], dtype=float)

    if var.ndim != 3:
        raise ValueError(f"{var_name} must have shape (Nt, Ny, Nx)")
    if var.shape[1] != lats.size or var.shape[2] != lons.size:
        raise ValueError("Grid dimensions do not match lats/lons")

    extracted = np.array(
        [
            _bilinear_on_regular_grid(lats, lons, var[t], station_lat, station_lon)
            for t in range(var.shape[0])
        ],
        dtype=float,
    )
    temps_f = _to_fahrenheit(extracted, units=units)
    times_local = _parse_times_to_local(times_utc, tz)
    return times_local, temps_f.astype(float).tolist()


def extract_station_series(
    npz_path: Path | None = None,
    station_lat: float | None = None,
    station_lon: float | None = None,
    tz: str | None = None,
) -> tuple[list[datetime], list[float]]:
    """Return station series from NPZ when provided, else deterministic fallback."""

    if npz_path is not None:
        if station_lat is None or station_lon is None:
            raise ValueError("station_lat and station_lon are required when npz_path is provided")
        timezone = tz or "America/Chicago"
        return extract_station_series_from_npz(
            npz_path=npz_path,
            station_lat=station_lat,
            station_lon=station_lon,
            tz=timezone,
        )

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

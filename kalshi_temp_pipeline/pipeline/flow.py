"""Smoke-test flow for MVP scaffold."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from kalshi_temp_pipeline.config import load_settings
from kalshi_temp_pipeline.pipeline.tasks.decision import make_decision
from kalshi_temp_pipeline.pipeline.tasks.mos import EmosModel
from kalshi_temp_pipeline.pipeline.tasks.obs_cli import ObsCliClient
from kalshi_temp_pipeline.pipeline.tasks.postproc import extract_station_series
from kalshi_temp_pipeline.pipeline.tasks.time_windows import (
    get_climate_day_window,
    tmax_over_window,
)
from kalshi_temp_pipeline.pipeline.tasks.verify import summarize_verification
from kalshi_temp_pipeline.pipeline.tasks.wrf import run_wrf


def _build_smoke_npz(path: str) -> None:
    lats = np.array([40.0, 41.0], dtype=float)
    lons = np.array([-74.0, -73.0], dtype=float)
    times_utc = np.array(["2026-07-10T06:00:00Z", "2026-07-10T12:00:00Z"], dtype="U25")

    t2m = np.array(
        [
            [[300.0, 301.0], [302.0, 303.0]],
            [[301.0, 302.0], [303.0, 304.0]],
        ],
        dtype=float,
    )
    np.savez(path, lats=lats, lons=lons, times_utc=times_utc, t2m=t2m)


def smoke_flow() -> str:
    """Run DRY_RUN-safe placeholder pipeline and return final status."""

    settings = load_settings()
    if not settings.dry_run:
        raise RuntimeError("Safety check failed: DRY_RUN must be true for smoke flow")

    _ = run_wrf()

    obs_client = ObsCliClient()
    _ = obs_client.fetch_daily_cli(station="KDSM", year=2026)

    with TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "smoke_grid.npz"
        _build_smoke_npz(str(npz_path))
        times, temps = extract_station_series(
            npz_path=npz_path,
            station_lat=40.5,
            station_lon=-73.5,
            tz=settings.timezone,
        )

    assert times[0].tzinfo == ZoneInfo(settings.timezone)

    window = get_climate_day_window(date(2026, 7, 10), settings.timezone)
    _ = tmax_over_window(times, temps, window)

    mos = EmosModel(mode="deterministic")
    mos.fit(pd.DataFrame({"x": [80.0, 82.0, 85.0], "y": [81.0, 83.0, 84.5]}))
    dist = mos.predict_distribution(pd.DataFrame({"x": [84.0]}))
    bins = [(-1e9, 82.0), (82.0, 86.0), (86.0, 1e9)]
    probs = mos.predict_bin_probs(pd.DataFrame({"x": [84.0]}), bins)

    orderbook = {"yes_price_cents": 61.0}
    _ = make_decision(
        probability=float(probs[0, 1]),
        price_cents=float(orderbook["yes_price_cents"]),
        edge_threshold=0.05,
        min_hit_rate=0.50,
        current_exposure_usd=0.0,
        order_size_usd=10.0,
        max_exposure_usd=100.0,
    )

    _ = dist
    _ = summarize_verification()

    print("OK")  # noqa: T201
    return "OK"


def main() -> None:
    smoke_flow()


if __name__ == "__main__":
    main()

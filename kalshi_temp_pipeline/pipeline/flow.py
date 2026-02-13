"""Prefect-based smoke flow for MVP scaffold."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from prefect import flow, task

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


@task
def task_run_model() -> dict[str, str]:
    return run_wrf()


@task(retries=2, retry_delay_seconds=1)
def task_fetch_obs() -> dict[str, object]:
    obs_client = ObsCliClient()
    return obs_client.fetch_daily_cli(station="KDSM", year=2026)


@task
def task_extract_station_series(tz: str) -> tuple[list[datetime], list[float]]:
    with TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "smoke_grid.npz"
        _build_smoke_npz(str(npz_path))
        return extract_station_series(
            npz_path=npz_path,
            station_lat=40.5,
            station_lon=-73.5,
            tz=tz,
        )


@task
def task_compute_tmax(times: list[datetime], temps: list[float], tz: str) -> float:
    if times[0].tzinfo != ZoneInfo(tz):
        raise RuntimeError("Unexpected timezone in extracted station series")
    window = get_climate_day_window(date(2026, 7, 10), tz)
    return tmax_over_window(times, temps, window)


@task
def task_mos_predict() -> tuple[np.ndarray, object]:
    mos = EmosModel(mode="deterministic")
    mos.fit(pd.DataFrame({"x": [80.0, 82.0, 85.0], "y": [81.0, 83.0, 84.5]}))
    dist = mos.predict_distribution(pd.DataFrame({"x": [84.0]}))
    bins = [(-1e9, 82.0), (82.0, 86.0), (86.0, 1e9)]
    probs = mos.predict_bin_probs(pd.DataFrame({"x": [84.0]}), bins)
    return probs, dist


@task(retries=2, retry_delay_seconds=1)
def task_kalshi_read() -> dict[str, float]:
    # DRY_RUN-safe local stub; task boundary mirrors production Kalshi read stage.
    return {"yes_price_cents": 61.0}


@task
def task_decision(probs: np.ndarray, orderbook: dict[str, float]) -> object:
    return make_decision(
        probability=float(probs[0, 1]),
        price_cents=float(orderbook["yes_price_cents"]),
        edge_threshold=0.05,
        min_hit_rate=0.50,
        current_exposure_usd=0.0,
        order_size_usd=10.0,
        max_exposure_usd=100.0,
    )


@task
def task_verify(probs: np.ndarray) -> dict[str, float]:
    return summarize_verification(probs, realized_bin_index=1)


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


@flow(name="kalshi-smoke")
def smoke_prefect_flow() -> str:
    """Run DRY_RUN-safe Prefect smoke flow and return final status."""

    settings = load_settings()
    if not settings.dry_run:
        raise RuntimeError("Safety check failed: DRY_RUN must be true for smoke flow")

    _ = task_run_model()
    _ = task_fetch_obs()
    times, temps = task_extract_station_series(settings.timezone)
    _ = task_compute_tmax(times, temps, settings.timezone)
    probs, _dist = task_mos_predict()
    orderbook = task_kalshi_read()
    _ = task_decision(probs, orderbook)
    _ = task_verify(probs)

    print("OK")  # noqa: T201
    return "OK"


def main() -> None:
    smoke_prefect_flow()


if __name__ == "__main__":
    main()

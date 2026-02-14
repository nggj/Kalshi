"""CLI for station-centric verification report generation (DRY_RUN-safe)."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    PublicModelBaselineCSV,
)
from kalshi_temp_pipeline.pipeline.tasks.distributions import NormalDist
from kalshi_temp_pipeline.pipeline.tasks.verify import evaluate_station, generate_station_report


def _synthetic_history(target_date: date) -> dict[date, float]:
    return {
        target_date.replace(day=max(1, target_date.day - 3)): 81.0,
        target_date.replace(day=max(1, target_date.day - 2)): 82.0,
        target_date.replace(day=max(1, target_date.day - 1)): 84.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--show-hier-bias", action="store_true")
    parser.add_argument(
        "--public-csv",
        default="artifacts/baselines/public_model_station.csv",
        help="CSV path with columns: date,station,tmax",
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    station = args.station

    observed_tmax = 83.0
    raw_model_tmax = 84.8
    mos_dist = NormalDist(mu=np.array([83.4]), sigma=np.array([1.5]))
    bins = [(-1e9, 80.0), (80.0, 85.0), (85.0, 1e9)]
    mos_bin_probs = np.array([[0.15, 0.70, 0.15]])

    history = _synthetic_history(target_date)
    baselines = {
        "baseline_persistence": PersistenceBaseline(history, window_days=1),
        "baseline_climatology": ClimatologyBaseline(history),
    }

    public_csv = Path(args.public_csv)
    if public_csv.exists():
        baselines["baseline_public_csv"] = PublicModelBaselineCSV(public_csv)

    metrics = evaluate_station(
        target_date=target_date,
        station=station,
        observed_tmax=observed_tmax,
        raw_model_tmax=raw_model_tmax,
        mos_dist=mos_dist,
        mos_bin_probs=mos_bin_probs,
        bins=bins,
        baselines=baselines,
    )

    report_path = generate_station_report(
        target_date=target_date,
        station=station,
        metrics=metrics,
        show_hier_bias=args.show_hier_bias,
    )
    print(report_path)  # noqa: T201


if __name__ == "__main__":
    main()

from datetime import date
from pathlib import Path

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.baselines import ClimatologyBaseline, PersistenceBaseline
from kalshi_temp_pipeline.pipeline.tasks.distributions import NormalDist
from kalshi_temp_pipeline.pipeline.tasks.verify import evaluate_station, generate_station_report


def test_evaluate_station_metrics_known_case(tmp_path: Path) -> None:
    target_date = date(2026, 2, 13)
    station = "KNYC"
    observed_tmax = 84.0
    raw_model_tmax = 86.0

    mos_dist = NormalDist(mu=np.array([84.0]), sigma=np.array([1.0]))
    mos_probs = np.array([[0.1, 0.8, 0.1]])
    bins = [(-1e9, 80.0), (80.0, 85.0), (85.0, 1e9)]

    history = {
        date(2026, 2, 10): 82.0,
        date(2026, 2, 11): 83.0,
        date(2026, 2, 12): 83.5,
    }
    baselines = {
        "persistence": PersistenceBaseline(history),
        "climatology": ClimatologyBaseline(history),
    }

    metrics = evaluate_station(
        target_date=target_date,
        station=station,
        observed_tmax=observed_tmax,
        raw_model_tmax=raw_model_tmax,
        mos_dist=mos_dist,
        mos_bin_probs=mos_probs,
        bins=bins,
        baselines=baselines,
    )

    assert metrics["mos_mean"].mae == 0.0
    assert metrics["raw_model_station"].mae == 2.0
    assert metrics["mos_mean"].brier <= metrics["raw_model_station"].brier

    report = generate_station_report(
        target_date=target_date,
        station=station,
        metrics=metrics,
        output_root=tmp_path / "reports",
    )
    text = report.read_text(encoding="utf-8")
    assert "Edge summary" in text
    assert "| Method | MAE | RMSE | Bias | Brier |" in text


def test_generate_report_includes_hier_bias_section_when_enabled(tmp_path: Path) -> None:
    target_date = date(2026, 2, 13)
    station = "KNYC"

    metrics = {
        "mos_mean": evaluate_station(
            target_date=target_date,
            station=station,
            observed_tmax=84.0,
            raw_model_tmax=84.0,
            mos_dist=NormalDist(mu=np.array([84.0]), sigma=np.array([1.0])),
            mos_bin_probs=np.array([[0.1, 0.8, 0.1]]),
            bins=[(-1e9, 80.0), (80.0, 85.0), (85.0, 1e9)],
            baselines={"persistence": PersistenceBaseline({date(2026, 2, 12): 83.0})},
        )["mos_mean"]
    }

    report = generate_station_report(
        target_date=target_date,
        station=station,
        metrics=metrics,
        output_root=tmp_path / "reports",
        show_hier_bias=True,
        station_bias_estimate=0.45,
        season_bias_estimate=-0.20,
    )
    text = report.read_text(encoding="utf-8")

    assert "## Hierarchical bias correction" in text
    assert "station_bias estimate: 0.450" in text
    assert "season_bias estimate (target date month=2): -0.200" in text

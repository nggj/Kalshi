from datetime import date
from pathlib import Path

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.baselines import ClimatologyBaseline, PersistenceBaseline
from kalshi_temp_pipeline.pipeline.tasks.mos import NormalDist
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

from datetime import date
from pathlib import Path

import pytest

from kalshi_temp_pipeline.pipeline.tasks.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    PublicModelBaselineCSV,
)


def test_persistence_baseline_latest_and_rolling() -> None:
    history = {
        date(2026, 2, 10): 40.0,
        date(2026, 2, 11): 42.0,
        date(2026, 2, 12): 44.0,
    }
    latest = PersistenceBaseline(history, window_days=1)
    rolling = PersistenceBaseline(history, window_days=2)

    assert latest.predict_tmax(date(2026, 2, 13), "KNYC") == 44.0
    assert rolling.predict_tmax(date(2026, 2, 13), "KNYC") == 43.0


def test_climatology_baseline_day_of_year() -> None:
    history = {
        date(2024, 2, 13): 50.0,
        date(2025, 2, 13): 54.0,
        date(2025, 2, 14): 70.0,
    }
    clim = ClimatologyBaseline(history)
    assert clim.predict_tmax(date(2026, 2, 13), "KNYC") == 52.0


def test_public_model_baseline_csv_fixture() -> None:
    csv_path = Path("tests/fixtures/public_baseline_sample.csv")
    baseline = PublicModelBaselineCSV(csv_path)
    got = baseline.predict_tmax(date(2026, 2, 13), "KNYC")
    assert got == 82.5


def test_public_model_baseline_missing_row() -> None:
    csv_path = Path("tests/fixtures/public_baseline_sample.csv")
    baseline = PublicModelBaselineCSV(csv_path)
    with pytest.raises(ValueError):
        baseline.predict_tmax(date(2026, 2, 14), "KNYC")

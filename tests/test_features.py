import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.features import build_daily_training_frame


def test_build_daily_training_frame_joins_on_date_station() -> None:
    obs_df = pd.DataFrame(
        {
            "date": ["2026-02-13", "2026-02-13", "2026-02-14"],
            "station": ["KNYC", "KBOS", "KNYC"],
            "tmax": [45.0, 38.0, 47.0],
        }
    )
    wrf_df = pd.DataFrame(
        {
            "date": ["2026-02-13", "2026-02-14", "2026-02-13"],
            "station": ["KNYC", "KNYC", "KORD"],
            "tmax": [44.2, 48.1, 30.0],
        }
    )

    out = build_daily_training_frame(obs_df=obs_df, wrf_df=wrf_df)

    assert list(out["station"]) == ["KNYC", "KNYC"]
    assert list(out["date"]) == ["2026-02-13", "2026-02-14"]
    assert list(out["y"]) == [45.0, 47.0]
    assert list(out["wrf_tmax"]) == [44.2, 48.1]


def test_seasonal_terms_are_in_unit_range() -> None:
    obs_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-06-15", "2026-12-31"],
            "station": ["KNYC", "KNYC", "KNYC"],
            "tmax": [32.0, 79.0, 40.0],
        }
    )
    wrf_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-06-15", "2026-12-31"],
            "station": ["KNYC", "KNYC", "KNYC"],
            "tmax": [31.5, 80.2, 39.2],
        }
    )

    out = build_daily_training_frame(obs_df=obs_df, wrf_df=wrf_df)

    assert np.all(out["doy_sin"].to_numpy(dtype=float) <= 1.0)
    assert np.all(out["doy_sin"].to_numpy(dtype=float) >= -1.0)
    assert np.all(out["doy_cos"].to_numpy(dtype=float) <= 1.0)
    assert np.all(out["doy_cos"].to_numpy(dtype=float) >= -1.0)


def test_missing_public_values_are_nan() -> None:
    obs_df = pd.DataFrame(
        {
            "date": ["2026-02-13", "2026-02-14"],
            "station": ["KNYC", "KNYC"],
            "tmax": [45.0, 47.0],
        }
    )
    wrf_df = pd.DataFrame(
        {
            "date": ["2026-02-13", "2026-02-14"],
            "station": ["KNYC", "KNYC"],
            "tmax": [44.2, 48.1],
        }
    )
    public_df = pd.DataFrame(
        {
            "date": ["2026-02-13"],
            "station": ["KNYC"],
            "tmax": [43.9],
        }
    )

    out = build_daily_training_frame(obs_df=obs_df, wrf_df=wrf_df, public_df=public_df)

    assert float(out.loc[out["date"] == "2026-02-13", "public_tmax"].iloc[0]) == 43.9
    assert np.isnan(float(out.loc[out["date"] == "2026-02-14", "public_tmax"].iloc[0]))

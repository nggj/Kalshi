"""Pure feature-building transforms for MOS training data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _standardize_columns(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    required = {"date", "station", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = df.loc[:, ["date", "station", value_col]].copy()
    out = out.rename(columns={value_col: out_col})
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["station"] = out["station"].astype(str)
    return out


def _seasonal_terms(date_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    date_dt = pd.to_datetime(date_series, errors="raise")
    doy = date_dt.dt.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * (doy / 365.25)
    return np.sin(angle), np.cos(angle)


def build_daily_training_frame(
    obs_df: pd.DataFrame,
    wrf_df: pd.DataFrame,
    public_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build station/date keyed MOS training frame.

    Output columns:
    - date (ISO string)
    - station
    - y
    - wrf_tmax
    - public_tmax (NaN when missing or when public_df is None)
    - doy_sin
    - doy_cos
    """

    obs = _standardize_columns(obs_df, value_col="tmax", out_col="y")
    wrf = _standardize_columns(wrf_df, value_col="tmax", out_col="wrf_tmax")

    merged = obs.merge(wrf, on=["date", "station"], how="inner", validate="one_to_one")

    if public_df is None:
        merged["public_tmax"] = np.nan
    else:
        public = _standardize_columns(public_df, value_col="tmax", out_col="public_tmax")
        merged = merged.merge(public, on=["date", "station"], how="left", validate="one_to_one")

    doy_sin, doy_cos = _seasonal_terms(merged["date"])
    merged["doy_sin"] = doy_sin
    merged["doy_cos"] = doy_cos

    cols: Sequence[str] = [
        "date",
        "station",
        "y",
        "wrf_tmax",
        "public_tmax",
        "doy_sin",
        "doy_cos",
    ]
    out = merged.loc[:, cols].sort_values(["date", "station"]).reset_index(drop=True)
    return out

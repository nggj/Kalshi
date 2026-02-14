"""Empirical-Bayes hierarchical bias correction for station and season."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _ShrinkageStats:
    tau2: float
    sigma2: float


class HierarchicalBiasCorrector:
    """Shrinkage bias corrector over station and month residual effects."""

    def __init__(self) -> None:
        self.station_bias: dict[str, float] = {}
        self.season_bias: dict[int, float] = {}
        self._station_stats: _ShrinkageStats | None = None
        self._season_stats: _ShrinkageStats | None = None

    def fit(self, df: pd.DataFrame) -> None:
        required = {"station", "date", "residual"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        work = df.loc[:, ["station", "date", "residual"]].copy()
        work["station"] = work["station"].astype(str)
        work["residual"] = work["residual"].to_numpy(dtype=float)
        work["month"] = pd.to_datetime(work["date"], errors="raise").dt.month

        self.station_bias, self._station_stats = self._fit_station_bias(work)

        station_adj = np.array(
            [self.station_bias.get(st, 0.0) for st in work["station"].to_numpy(dtype=str)],
            dtype=float,
        )
        work_month = work.copy()
        work_month["residual"] = work_month["residual"].to_numpy(dtype=float) - station_adj
        self.season_bias, self._season_stats = self._fit_month_bias(work_month)

    def _group_stats(
        self, df: pd.DataFrame, key: str
    ) -> tuple[pd.Series, pd.Series, _ShrinkageStats]:
        grouped = df.groupby(key)["residual"]
        means = grouped.mean()
        counts = grouped.size().astype(float)

        sigma2 = float(df["residual"].var(ddof=1))
        if not np.isfinite(sigma2) or sigma2 <= 0.0:
            sigma2 = 1e-6

        between = float(means.var(ddof=1)) if len(means) > 1 else 0.0
        sampling_noise = float(np.mean(sigma2 / np.maximum(counts.to_numpy(dtype=float), 1.0)))
        tau2 = max(between - sampling_noise, 1e-6)
        return means, counts, _ShrinkageStats(tau2=tau2, sigma2=sigma2)

    def _fit_station_bias(self, df: pd.DataFrame) -> tuple[dict[str, float], _ShrinkageStats]:
        means, counts, stats = self._group_stats(df, "station")
        shrink = counts / (counts + (stats.sigma2 / stats.tau2))
        posterior = shrink * means
        out = {str(k): float(v) for k, v in posterior.to_dict().items()}
        return out, stats

    def _fit_month_bias(self, df: pd.DataFrame) -> tuple[dict[int, float], _ShrinkageStats]:
        means, counts, stats = self._group_stats(df, "month")
        shrink = counts / (counts + (stats.sigma2 / stats.tau2))
        posterior = shrink * means
        out = {int(k): float(v) for k, v in posterior.to_dict().items()}
        return out, stats

    def apply(
        self,
        mu: np.ndarray,
        station: np.ndarray | list[str],
        date: np.ndarray | list[str] | pd.Series,
    ) -> np.ndarray:
        if self._station_stats is None or self._season_stats is None:
            raise RuntimeError("HierarchicalBiasCorrector is not fitted")

        mu_arr = np.asarray(mu, dtype=float)
        station_arr = np.asarray(station, dtype=str)
        date_month = pd.to_datetime(pd.Series(date), errors="raise").dt.month.to_numpy(dtype=int)

        if mu_arr.shape[0] != station_arr.shape[0] or mu_arr.shape[0] != date_month.shape[0]:
            raise ValueError("mu, station, date must have the same length")

        station_adj = np.array([self.station_bias.get(s, 0.0) for s in station_arr], dtype=float)
        season_adj = np.array([self.season_bias.get(m, 0.0) for m in date_month], dtype=float)

        return cast(np.ndarray, mu_arr + station_adj + season_adj)

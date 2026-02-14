"""Nowcast helpers for max-so-far conditioned Tmax distributions."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.distributions import (
    NormalDist,
    PredictiveDistribution,
    condition_on_minimum,
)


def max_so_far(times: list[datetime], temps: list[float], as_of: datetime) -> float:
    """Return max observed temperature up to as_of (inclusive)."""

    if len(times) != len(temps):
        raise ValueError("times and temps must have same length")

    eligible = [float(t) for ts, t in zip(times, temps) if ts <= as_of]
    if not eligible:
        raise ValueError("no observations available at or before as_of")
    return float(max(eligible))


def nowcast_bin_probs(
    dist: PredictiveDistribution, bins: list[tuple[float, float]], max_so_far_value: float
) -> np.ndarray:
    """Compute bin probabilities for distribution conditioned on Tmax >= max_so_far."""

    conditioned = condition_on_minimum(dist, max_so_far_value)
    mean_shape = conditioned.mean().shape

    lows = np.array([b[0] for b in bins], dtype=float)
    highs = np.array([b[1] for b in bins], dtype=float)

    cdf_high = np.column_stack([conditioned.cdf(np.full(mean_shape, h)) for h in highs])
    cdf_low = np.column_stack([conditioned.cdf(np.full(mean_shape, low)) for low in lows])
    probs = np.clip(cdf_high - cdf_low, 0.0, 1.0)

    row_sums = probs.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return cast(np.ndarray, probs / safe)


class NowcastLinearUpdater:
    """Lead-time bucketed linear updater for mu/sigma."""

    def __init__(self, near_cutoff_hours: float = 6.0) -> None:
        self.near_cutoff_hours = near_cutoff_hours
        self.k_by_bucket: dict[str, float] = {}
        self.shrink_by_bucket: dict[str, float] = {}

    def _bucket(self, lead_hours: np.ndarray) -> np.ndarray:
        return np.where(lead_hours <= self.near_cutoff_hours, "near", "far")

    def fit(self, df: pd.DataFrame) -> None:
        required = {"lead_hours", "mu_base", "sigma_base", "temp_obs_now", "temp_fcst_now", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        lead = df["lead_hours"].to_numpy(dtype=float)
        mu_base = df["mu_base"].to_numpy(dtype=float)
        sigma_base = np.maximum(df["sigma_base"].to_numpy(dtype=float), 1e-6)
        now_obs = df["temp_obs_now"].to_numpy(dtype=float)
        now_fcst = df["temp_fcst_now"].to_numpy(dtype=float)
        now_err = now_obs - now_fcst
        resid = df["y"].to_numpy(dtype=float) - mu_base
        buckets = self._bucket(lead)

        self.k_by_bucket = {}
        self.shrink_by_bucket = {}

        for bucket in ("near", "far"):
            mask = buckets == bucket
            if not np.any(mask):
                self.k_by_bucket[bucket] = 0.0
                self.shrink_by_bucket[bucket] = 1.0
                continue

            x = now_err[mask]
            y = resid[mask]
            x_var = float(np.var(x))
            if x_var <= 1e-12:
                k = 0.0
            else:
                k = float(np.cov(x, y, ddof=0)[0, 1] / x_var)

            resid_after = y - k * x
            shrink = float(np.std(resid_after) / np.mean(sigma_base[mask]))
            shrink = float(np.clip(shrink, 0.1, 2.0))

            self.k_by_bucket[bucket] = k
            self.shrink_by_bucket[bucket] = shrink

    def update_distribution(
        self,
        dist: PredictiveDistribution,
        lead_hours: np.ndarray,
        temp_obs_now: np.ndarray,
        temp_fcst_now: np.ndarray,
    ) -> PredictiveDistribution:
        if not self.k_by_bucket or not self.shrink_by_bucket:
            raise RuntimeError("NowcastLinearUpdater is not fitted")
        if not isinstance(dist, NormalDist):
            raise TypeError("NowcastLinearUpdater currently supports NormalDist only")

        lead = np.asarray(lead_hours, dtype=float)
        obs = np.asarray(temp_obs_now, dtype=float)
        fcst = np.asarray(temp_fcst_now, dtype=float)
        if lead.shape[0] != dist.mu.shape[0] or obs.shape[0] != dist.mu.shape[0]:
            raise ValueError("lead_hours/temp_obs_now/temp_fcst_now must match distribution length")

        buckets = self._bucket(lead)
        now_err = obs - fcst

        k = np.array([self.k_by_bucket.get(b, 0.0) for b in buckets], dtype=float)
        shrink = np.array([self.shrink_by_bucket.get(b, 1.0) for b in buckets], dtype=float)

        mu_adj = dist.mu + k * now_err
        sigma_adj = np.maximum(dist.sigma * shrink, 1e-6)
        return NormalDist(mu=mu_adj, sigma=sigma_adj)

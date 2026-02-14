"""Stacked Gaussian MOS using WRF + Public predictors."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from kalshi_temp_pipeline.pipeline.tasks.distributions import NormalDist


def _softplus(x: np.ndarray) -> np.ndarray:
    out = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return cast(np.ndarray, out)


def _inverse_softplus(y: float) -> float:
    y_safe = max(y, 1e-6)
    return float(np.log(np.expm1(y_safe)))


class StackedGaussianMosModel:
    """Gaussian MOS model with stacked mean predictors and positive sigma."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._mu_model: Ridge | None = None
        self._sigma_raw: float | None = None
        self._public_mean: float | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        required = {"wrf_tmax", "public_tmax", "doy_sin", "doy_cos", "y"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        public = train_df["public_tmax"].to_numpy(dtype=float)
        if np.all(np.isnan(public)):
            self._public_mean = 0.0
        elif np.isnan(public).any():
            self._public_mean = float(np.nanmean(public))
        else:
            self._public_mean = float(np.mean(public))
        public_filled = np.where(np.isnan(public), self._public_mean, public)

        x = np.column_stack(
            [
                train_df["wrf_tmax"].to_numpy(dtype=float),
                public_filled,
                train_df["doy_sin"].to_numpy(dtype=float),
                train_df["doy_cos"].to_numpy(dtype=float),
            ]
        )
        y = train_df["y"].to_numpy(dtype=float)

        model = Ridge(alpha=self.alpha, random_state=0)
        model.fit(x, y)
        pred = model.predict(x)
        resid_std = float(np.std(y - pred))

        self._mu_model = model
        self._sigma_raw = _inverse_softplus(resid_std)

    def predict_distribution(self, features_df: pd.DataFrame) -> NormalDist:
        if self._mu_model is None or self._sigma_raw is None or self._public_mean is None:
            raise RuntimeError("StackedGaussianMosModel is not fitted")

        required = {"wrf_tmax", "public_tmax", "doy_sin", "doy_cos"}
        missing = required - set(features_df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        public = features_df["public_tmax"].to_numpy(dtype=float)
        public_filled = np.where(np.isnan(public), self._public_mean, public)
        x = np.column_stack(
            [
                features_df["wrf_tmax"].to_numpy(dtype=float),
                public_filled,
                features_df["doy_sin"].to_numpy(dtype=float),
                features_df["doy_cos"].to_numpy(dtype=float),
            ]
        )

        mu = cast(np.ndarray, self._mu_model.predict(x))
        sigma = _softplus(np.full_like(mu, self._sigma_raw, dtype=float)) + 1e-6
        return NormalDist(mu=mu, sigma=sigma)

    def predict_bin_probs(
        self, features_df: pd.DataFrame, bins: list[tuple[float, float]]
    ) -> np.ndarray:
        dist = self.predict_distribution(features_df)
        lows = np.array([b[0] for b in bins], dtype=float)
        highs = np.array([b[1] for b in bins], dtype=float)

        mean_shape = dist.mean().shape
        cdf_high = np.column_stack([dist.cdf(np.full(mean_shape, high)) for high in highs])
        cdf_low = np.column_stack([dist.cdf(np.full(mean_shape, low)) for low in lows])
        probs = np.clip(cdf_high - cdf_low, 0.0, 1.0)

        row_sums = probs.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        return cast(np.ndarray, probs / safe_sums)

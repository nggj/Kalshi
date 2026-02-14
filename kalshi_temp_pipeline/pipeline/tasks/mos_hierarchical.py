"""Hierarchical station/season bias correction wrapper for MOS models."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.distributions import (
    MixtureNormalDist,
    NormalDist,
    PredictiveDistribution,
    QuantileDist,
    SkewNormalDist,
)
from kalshi_temp_pipeline.pipeline.tasks.hierarchical import HierarchicalBiasCorrector


class _BaseMosModel(Protocol):
    def fit(self, train_df: pd.DataFrame) -> None: ...

    def predict_distribution(self, features_df: pd.DataFrame) -> PredictiveDistribution: ...


class HierarchicalCorrectedMosModel:
    """Wraps a MOS model and applies hierarchical station/season mean correction."""

    def __init__(
        self, base_model: _BaseMosModel, bias_corrector: HierarchicalBiasCorrector
    ) -> None:
        self.base_model = base_model
        self.bias_corrector = bias_corrector

    def fit(self, train_df: pd.DataFrame) -> None:
        required = {"station", "date", "y"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        self.base_model.fit(train_df)
        base_dist = self.base_model.predict_distribution(train_df)
        mu = base_dist.mean()
        y = train_df["y"].to_numpy(dtype=float)

        fit_df = pd.DataFrame(
            {
                "station": train_df["station"].astype(str).to_numpy(),
                "date": train_df["date"].to_numpy(),
                "residual": y - mu,
            }
        )
        self.bias_corrector.fit(fit_df)

    def predict_distribution(self, features_df: pd.DataFrame) -> PredictiveDistribution:
        required = {"station", "date"}
        missing = required - set(features_df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        base_dist = self.base_model.predict_distribution(features_df)
        base_mean = base_dist.mean()
        adjusted_mean = self.bias_corrector.apply(
            mu=base_mean,
            station=features_df["station"].astype(str).to_numpy(),
            date=features_df["date"],
        )
        delta = adjusted_mean - base_mean
        return self._shift_distribution(base_dist, delta)

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

        sums = probs.sum(axis=1, keepdims=True)
        safe_sums = np.where(sums > 0, sums, 1.0)
        return cast(np.ndarray, probs / safe_sums)

    def _shift_distribution(
        self, dist: PredictiveDistribution, delta: np.ndarray
    ) -> PredictiveDistribution:
        if isinstance(dist, NormalDist):
            return NormalDist(mu=dist.mu + delta, sigma=dist.sigma)
        if isinstance(dist, QuantileDist):
            return QuantileDist(q_levels=dist.q_levels, q_values=dist.q_values + delta[:, None])
        if isinstance(dist, MixtureNormalDist):
            return MixtureNormalDist(
                weights=dist.weights,
                mus=dist.mus + delta[:, None],
                sigmas=dist.sigmas,
            )
        if isinstance(dist, SkewNormalDist):
            return SkewNormalDist(loc=dist.loc + delta, scale=dist.scale, shape=dist.shape)
        raise TypeError(f"Unsupported distribution type for hierarchical correction: {type(dist)}")

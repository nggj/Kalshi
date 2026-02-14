"""Quantile-regression MOS model for non-normal predictive distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from kalshi_temp_pipeline.pipeline.tasks.distributions import QuantileDist


@dataclass(frozen=True)
class _QuantileModelBundle:
    quantile: float
    model: QuantileRegressor


class QuantileRegressionMosModel:
    """MOS model using per-quantile linear regressors."""

    def __init__(self, mode: str = "auto", q_levels: np.ndarray | None = None) -> None:
        self.mode = mode
        self.q_levels = q_levels if q_levels is not None else np.linspace(0.05, 0.95, 19)
        self._resolved_mode: str | None = None
        self._models: list[_QuantileModelBundle] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit quantile regressors.

        deterministic mode expects columns: x, y
        ensemble mode expects columns: ens_mean, ens_spread, y
        """

        resolved = self._resolve_mode(train_df)
        y = train_df["y"].to_numpy(dtype=float)
        x_mat = self._design_matrix(train_df, resolved)

        self._models = []
        for q in self.q_levels:
            reg = QuantileRegressor(
                quantile=float(q),
                alpha=0.0,
                fit_intercept=True,
                solver="highs",
            )
            reg.fit(x_mat, y)
            self._models.append(_QuantileModelBundle(quantile=float(q), model=reg))

        self._resolved_mode = resolved

    def predict_distribution(self, features_df: pd.DataFrame) -> QuantileDist:
        """Predict quantile-based distribution."""

        if not self._models or self._resolved_mode is None:
            raise RuntimeError("QuantileRegressionMosModel is not fitted")

        x_mat = self._design_matrix(features_df, self._resolved_mode)
        preds = np.column_stack([bundle.model.predict(x_mat) for bundle in self._models])
        preds_monotone = np.maximum.accumulate(preds, axis=1)
        return QuantileDist(q_levels=self.q_levels, q_values=preds_monotone)

    def predict_bin_probs(
        self, features_df: pd.DataFrame, bins: list[tuple[float, float]]
    ) -> np.ndarray:
        """Predict bin probabilities from quantile-based CDF differences."""

        dist = self.predict_distribution(features_df)
        mean_shape = dist.mean().shape

        lows = np.array([b[0] for b in bins], dtype=float)
        highs = np.array([b[1] for b in bins], dtype=float)

        cdf_high = np.column_stack([dist.cdf(np.full(mean_shape, high)) for high in highs])
        cdf_low = np.column_stack([dist.cdf(np.full(mean_shape, low)) for low in lows])
        probs = np.clip(cdf_high - cdf_low, 0.0, 1.0)

        row_sums = probs.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        return cast(np.ndarray, probs / safe_sums)

    def _resolve_mode(self, df: pd.DataFrame) -> str:
        if self.mode in {"deterministic", "ensemble"}:
            return self.mode
        if {"ens_mean", "ens_spread"}.issubset(df.columns):
            return "ensemble"
        if "x" in df.columns:
            return "deterministic"
        raise ValueError("Cannot infer mode: expected x or ens_mean/ens_spread columns")

    def _design_matrix(self, df: pd.DataFrame, mode: str) -> np.ndarray:
        if mode == "deterministic":
            return cast(np.ndarray, df[["x"]].to_numpy(dtype=float))
        return cast(np.ndarray, df[["ens_mean", "ens_spread"]].to_numpy(dtype=float))

"""Predictive distribution abstractions for MOS outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.stats import norm


class PredictiveDistribution(Protocol):
    """Protocol for predictive distributions used by MOS and verification."""

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Vectorized CDF, returning shape (N,) when x shape is (N,)."""

    def mean(self) -> np.ndarray:
        """Return per-sample predictive means with shape (N,)."""


@dataclass(frozen=True)
class NormalDist(PredictiveDistribution):
    """Normal predictive distribution parameterized by mu/sigma arrays."""

    mu: np.ndarray
    sigma: np.ndarray

    def cdf(self, x: np.ndarray) -> np.ndarray:
        cdf_vals = norm.cdf(x, loc=self.mu, scale=self.sigma)
        return cast(np.ndarray, cdf_vals)

    def mean(self) -> np.ndarray:
        return self.mu


@dataclass(frozen=True)
class QuantileDist(PredictiveDistribution):
    """Distribution represented by quantile levels and predicted quantile values."""

    q_levels: np.ndarray  # shape (Q,)
    q_values: np.ndarray  # shape (N,Q)

    def mean(self) -> np.ndarray:
        """Approximate E[X] by integrating quantile function over p in [0,1]."""

        if self.q_levels.ndim != 1 or self.q_values.ndim != 2:
            raise ValueError("q_levels must be 1D and q_values must be 2D")
        if self.q_values.shape[1] != self.q_levels.shape[0]:
            raise ValueError("q_values second dimension must match q_levels length")

        return cast(np.ndarray, np.trapezoid(self.q_values, self.q_levels, axis=1))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Approximate CDF by inverting the quantile function with interpolation."""

        if x.ndim != 1:
            raise ValueError("x must be 1D")
        if self.q_values.shape[0] != x.shape[0]:
            raise ValueError("x length must match number of rows in q_values")

        out = np.zeros_like(x, dtype=float)
        for i in range(x.shape[0]):
            qv = np.asarray(self.q_values[i], dtype=float)
            # enforce monotone quantile curve for stable inversion
            qv_monotone = np.maximum.accumulate(qv)

            if x[i] < qv_monotone[0]:
                out[i] = 0.0
            elif x[i] > qv_monotone[-1]:
                out[i] = 1.0
            else:
                out[i] = float(np.interp(x[i], qv_monotone, self.q_levels))

        return out

"""Predictive distribution abstractions for MOS outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.stats import norm, skewnorm


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
class TruncatedNormalDist(PredictiveDistribution):
    """Lower-truncated normal distribution."""

    mu: np.ndarray
    sigma: np.ndarray
    lower: np.ndarray

    def cdf(self, x: np.ndarray) -> np.ndarray:
        base_at_x = norm.cdf(x, loc=self.mu, scale=self.sigma)
        base_at_lower = norm.cdf(self.lower, loc=self.mu, scale=self.sigma)
        tail = np.maximum(1.0 - base_at_lower, 1e-12)
        out = np.where(x < self.lower, 0.0, (base_at_x - base_at_lower) / tail)
        return np.clip(out, 0.0, 1.0)

    def mean(self) -> np.ndarray:
        alpha = (self.lower - self.mu) / np.maximum(self.sigma, 1e-12)
        z = np.maximum(1.0 - norm.cdf(alpha), 1e-12)
        adj = norm.pdf(alpha) / z
        return cast(np.ndarray, self.mu + self.sigma * adj)


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
            qv_monotone = np.maximum.accumulate(qv)

            if x[i] < qv_monotone[0]:
                out[i] = 0.0
            elif x[i] > qv_monotone[-1]:
                out[i] = 1.0
            else:
                out[i] = float(np.interp(x[i], qv_monotone, self.q_levels))

        return out


@dataclass(frozen=True)
class MixtureNormalDist(PredictiveDistribution):
    """Two-component normal mixture distribution."""

    weights: np.ndarray  # shape (N,2)
    mus: np.ndarray  # shape (N,2)
    sigmas: np.ndarray  # shape (N,2)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError("x must be 1D")
        if self.weights.shape != self.mus.shape or self.weights.shape != self.sigmas.shape:
            raise ValueError("weights, mus, sigmas must have identical shapes")
        if self.weights.shape[1] != 2:
            raise ValueError("mixture inputs must have shape (N,2)")
        if self.weights.shape[0] != x.shape[0]:
            raise ValueError("x length must match number of samples")

        comp_cdf = norm.cdf(x[:, None], loc=self.mus, scale=self.sigmas)
        return cast(np.ndarray, np.sum(self.weights * comp_cdf, axis=1))

    def mean(self) -> np.ndarray:
        if self.weights.shape != self.mus.shape:
            raise ValueError("weights and mus must have identical shapes")
        return cast(np.ndarray, np.sum(self.weights * self.mus, axis=1))


@dataclass(frozen=True)
class TruncatedMixtureNormalDist(PredictiveDistribution):
    """Lower-truncated two-component normal mixture."""

    weights: np.ndarray
    mus: np.ndarray
    sigmas: np.ndarray
    lower: np.ndarray

    def cdf(self, x: np.ndarray) -> np.ndarray:
        base = MixtureNormalDist(weights=self.weights, mus=self.mus, sigmas=self.sigmas)
        base_x = base.cdf(x)
        base_l = base.cdf(self.lower)
        tail = np.maximum(1.0 - base_l, 1e-12)
        out = np.where(x < self.lower, 0.0, (base_x - base_l) / tail)
        return np.clip(out, 0.0, 1.0)

    def mean(self) -> np.ndarray:
        alpha = (self.lower[:, None] - self.mus) / np.maximum(self.sigmas, 1e-12)
        surv = np.maximum(1.0 - norm.cdf(alpha), 1e-12)
        comp_mass = self.weights * surv
        norm_mass = np.maximum(np.sum(comp_mass, axis=1, keepdims=True), 1e-12)
        post_w = comp_mass / norm_mass

        comp_mean = self.mus + self.sigmas * (norm.pdf(alpha) / surv)
        return cast(np.ndarray, np.sum(post_w * comp_mean, axis=1))


@dataclass(frozen=True)
class SkewNormalDist(PredictiveDistribution):
    """Skew-normal predictive distribution."""

    loc: np.ndarray  # shape (N,)
    scale: np.ndarray  # shape (N,)
    shape: np.ndarray  # shape (N,)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        cdf_vals = skewnorm.cdf(x, a=self.shape, loc=self.loc, scale=self.scale)
        return cast(np.ndarray, cdf_vals)

    def mean(self) -> np.ndarray:
        delta = self.shape / np.sqrt(1.0 + self.shape**2)
        return cast(np.ndarray, self.loc + self.scale * delta * np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class TruncatedSkewNormalDist(PredictiveDistribution):
    """Lower-truncated skew-normal distribution."""

    loc: np.ndarray
    scale: np.ndarray
    shape: np.ndarray
    lower: np.ndarray

    def cdf(self, x: np.ndarray) -> np.ndarray:
        base_x = skewnorm.cdf(x, a=self.shape, loc=self.loc, scale=self.scale)
        base_l = skewnorm.cdf(self.lower, a=self.shape, loc=self.loc, scale=self.scale)
        tail = np.maximum(1.0 - base_l, 1e-12)
        out = np.where(x < self.lower, 0.0, (base_x - base_l) / tail)
        return np.clip(out, 0.0, 1.0)

    def mean(self) -> np.ndarray:
        out = np.zeros_like(self.loc, dtype=float)
        for i in range(self.loc.shape[0]):
            lo = float(self.lower[i])
            hi = float(self.loc[i] + 12.0 * self.scale[i])
            if hi <= lo:
                hi = lo + max(float(self.scale[i]), 1.0)
            grid = np.linspace(lo, hi, 1000)
            pdf = skewnorm.pdf(grid, a=self.shape[i], loc=self.loc[i], scale=self.scale[i])
            mass = float(np.trapezoid(pdf, grid))
            if mass <= 1e-12:
                out[i] = lo
            else:
                out[i] = float(np.trapezoid(grid * pdf, grid) / mass)
        return out


def condition_on_minimum(dist: PredictiveDistribution, lower: float) -> PredictiveDistribution:
    """Condition distribution on X >= lower via CDF renormalization."""

    if np.isneginf(lower):
        return dist

    if isinstance(dist, NormalDist):
        lower_arr = np.full_like(dist.mu, lower, dtype=float)
        return TruncatedNormalDist(mu=dist.mu, sigma=dist.sigma, lower=lower_arr)

    if isinstance(dist, QuantileDist):
        n, q = dist.q_values.shape
        lower_vec = np.full(n, lower, dtype=float)
        f_lower = dist.cdf(lower_vec)
        tail = np.maximum(1.0 - f_lower, 1e-12)

        u = np.clip(f_lower[:, None] + tail[:, None] * dist.q_levels[None, :], 0.0, 1.0)
        new_q = np.zeros((n, q), dtype=float)
        for i in range(n):
            qv = np.maximum.accumulate(dist.q_values[i])
            new_q[i] = np.interp(u[i], dist.q_levels, qv, left=qv[0], right=qv[-1])
            new_q[i] = np.maximum(new_q[i], lower)
        return QuantileDist(q_levels=dist.q_levels, q_values=new_q)

    if isinstance(dist, MixtureNormalDist):
        lower_arr = np.full(dist.weights.shape[0], lower, dtype=float)
        return TruncatedMixtureNormalDist(
            weights=dist.weights,
            mus=dist.mus,
            sigmas=dist.sigmas,
            lower=lower_arr,
        )

    if isinstance(dist, SkewNormalDist):
        lower_arr = np.full_like(dist.loc, lower, dtype=float)
        return TruncatedSkewNormalDist(
            loc=dist.loc,
            scale=dist.scale,
            shape=dist.shape,
            lower=lower_arr,
        )

    raise TypeError(f"Unsupported distribution type for truncation: {type(dist)}")

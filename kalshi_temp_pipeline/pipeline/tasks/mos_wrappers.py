"""Optional residual-shape wrappers over EMOS distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew

from kalshi_temp_pipeline.pipeline.tasks.distributions import (
    MixtureNormalDist,
    SkewNormalDist,
)
from kalshi_temp_pipeline.pipeline.tasks.mos import EmosModel


@dataclass(frozen=True)
class ResidualMixtureParams:
    w: float
    s1: float
    s2: float


class ResidualMixtureCalibrator:
    """Learns a 2-normal mixture on standardized residuals."""

    def __init__(self) -> None:
        self._params: ResidualMixtureParams | None = None

    def fit(self, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> None:
        z = (y - mu) / np.maximum(sigma, 1e-6)

        def objective(theta: np.ndarray) -> float:
            w = 1.0 / (1.0 + np.exp(-theta[0]))
            s1 = np.exp(theta[1])
            s2 = np.exp(theta[2])
            density = (1.0 - w) * norm.pdf(z, loc=0.0, scale=s1) + w * norm.pdf(
                z, loc=0.0, scale=s2
            )
            return float(-np.sum(np.log(np.maximum(density, 1e-12))))

        init = np.array([0.0, np.log(0.8), np.log(1.4)], dtype=float)
        res = minimize(objective, init, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"Mixture calibrator failed to fit: {res.message}")

        w = 1.0 / (1.0 + np.exp(-res.x[0]))
        s1 = float(np.exp(res.x[1]))
        s2 = float(np.exp(res.x[2]))

        # normalize to unit variance in z-space
        mix_var = (1.0 - w) * (s1**2) + w * (s2**2)
        norm_factor = np.sqrt(max(mix_var, 1e-9))
        s1 /= norm_factor
        s2 /= norm_factor

        self._params = ResidualMixtureParams(w=float(w), s1=s1, s2=s2)

    def apply(self, mu: np.ndarray, sigma: np.ndarray) -> MixtureNormalDist:
        if self._params is None:
            raise RuntimeError("Mixture calibrator is not fitted")

        w = self._params.w
        s1 = self._params.s1
        s2 = self._params.s2
        n = mu.shape[0]

        weights = np.column_stack([np.full(n, 1.0 - w), np.full(n, w)])
        mus = np.column_stack([mu, mu])
        sigmas = np.column_stack([sigma * s1, sigma * s2])
        return MixtureNormalDist(weights=weights, mus=mus, sigmas=np.maximum(sigmas, 1e-6))


@dataclass(frozen=True)
class ResidualSkewParams:
    shape: float


class ResidualSkewCalibrator:
    """Learns a global skew-normal shape on standardized residuals."""

    def __init__(self) -> None:
        self._params: ResidualSkewParams | None = None

    def fit(self, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> None:
        z = (y - mu) / np.maximum(sigma, 1e-6)
        target = float(np.clip(skew(z, bias=False), -0.95, 0.95))

        def skew_from_shape(a: float) -> float:
            delta = a / np.sqrt(1.0 + a**2)
            num = (4.0 - np.pi) / 2.0 * (delta * np.sqrt(2.0 / np.pi)) ** 3
            den = (1.0 - 2.0 * delta**2 / np.pi) ** 1.5
            return float(num / max(den, 1e-12))

        lo, hi = -20.0, 20.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if skew_from_shape(mid) < target:
                lo = mid
            else:
                hi = mid
        self._params = ResidualSkewParams(shape=float((lo + hi) / 2.0))

    def apply(self, mu: np.ndarray, sigma: np.ndarray) -> SkewNormalDist:
        if self._params is None:
            raise RuntimeError("Skew calibrator is not fitted")

        a = self._params.shape
        delta = a / np.sqrt(1.0 + a**2)
        z_var = 1.0 - 2.0 * delta**2 / np.pi
        z_scale = 1.0 / np.sqrt(max(z_var, 1e-12))
        z_loc = -z_scale * delta * np.sqrt(2.0 / np.pi)

        scale = np.maximum(sigma * z_scale, 1e-6)
        loc = mu + sigma * z_loc
        shape = np.full_like(mu, a, dtype=float)
        return SkewNormalDist(loc=loc, scale=scale, shape=shape)


class MixtureWrappedEmosModel:
    """EMOS mean/scale wrapped by residual 2-normal mixture shape."""

    def __init__(self, mode: str = "auto") -> None:
        self.base = EmosModel(mode=mode)
        self.calibrator = ResidualMixtureCalibrator()

    def fit(self, train_df: pd.DataFrame) -> None:
        self.base.fit(train_df)
        base_dist = self.base.predict_distribution(train_df)
        y = train_df["y"].to_numpy(dtype=float)
        self.calibrator.fit(base_dist.mu, base_dist.sigma, y)

    def predict_distribution(self, features_df: pd.DataFrame) -> MixtureNormalDist:
        base_dist = self.base.predict_distribution(features_df)
        return self.calibrator.apply(base_dist.mu, base_dist.sigma)

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


class SkewWrappedEmosModel:
    """EMOS mean/scale wrapped by residual skew-normal shape."""

    def __init__(self, mode: str = "auto") -> None:
        self.base = EmosModel(mode=mode)
        self.calibrator = ResidualSkewCalibrator()

    def fit(self, train_df: pd.DataFrame) -> None:
        self.base.fit(train_df)
        base_dist = self.base.predict_distribution(train_df)
        y = train_df["y"].to_numpy(dtype=float)
        self.calibrator.fit(base_dist.mu, base_dist.sigma, y)

    def predict_distribution(self, features_df: pd.DataFrame) -> SkewNormalDist:
        base_dist = self.base.predict_distribution(features_df)
        return self.calibrator.apply(base_dist.mu, base_dist.sigma)

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

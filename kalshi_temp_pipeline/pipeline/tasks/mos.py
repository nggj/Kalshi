"""MOS/Calibration core for calibrated bin probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class MosOutput:
    """Backward-compatible output wrapper used by smoke flow."""

    probability: float


@dataclass(frozen=True)
class NormalDist:
    """Predicted normal distribution parameters."""

    mu: np.ndarray
    sigma: np.ndarray


def _softplus(x: np.ndarray) -> np.ndarray:
    out = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return cast(np.ndarray, out)


def _inverse_softplus(y: float) -> float:
    y_safe = max(y, 1e-6)
    return float(np.log(np.expm1(y_safe)))


class EmosModel:
    """EMOS-style model for deterministic or ensemble inputs."""

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode
        self._mu_coef: np.ndarray | None = None
        self._sigma_coef: np.ndarray | None = None
        self._resolved_mode: str | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit EMOS parameters from training dataframe.

        deterministic mode expects columns: x, y
        ensemble mode expects columns: ens_mean, ens_spread, y
        """

        resolved = self._resolve_mode(train_df)
        y = train_df["y"].to_numpy(dtype=float)

        if resolved == "deterministic":
            x = train_df["x"].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            self._mu_coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design @ self._mu_coef
            resid_std = float(np.std(resid))
            self._sigma_coef = np.array([_inverse_softplus(resid_std)])
        else:
            ens_mean = train_df["ens_mean"].to_numpy(dtype=float)
            ens_spread = train_df["ens_spread"].to_numpy(dtype=float)
            mu_design = np.column_stack([np.ones(len(ens_mean)), ens_mean])
            self._mu_coef, *_ = np.linalg.lstsq(mu_design, y, rcond=None)
            resid = np.abs(y - mu_design @ self._mu_coef)
            sigma_design = np.column_stack([np.ones(len(ens_spread)), ens_spread])
            raw_coef, *_ = np.linalg.lstsq(sigma_design, resid, rcond=None)
            self._sigma_coef = raw_coef

        self._resolved_mode = resolved

    def predict_distribution(self, features_df: pd.DataFrame) -> NormalDist:
        """Predict normal distribution parameters (mu/sigma)."""

        self._assert_fitted()
        assert self._resolved_mode is not None
        assert self._mu_coef is not None
        assert self._sigma_coef is not None

        if self._resolved_mode == "deterministic":
            x = features_df["x"].to_numpy(dtype=float)
            mu = self._mu_coef[0] + self._mu_coef[1] * x
            raw_sigma = np.full_like(mu, self._sigma_coef[0], dtype=float)
        else:
            ens_mean = features_df["ens_mean"].to_numpy(dtype=float)
            ens_spread = features_df["ens_spread"].to_numpy(dtype=float)
            mu = self._mu_coef[0] + self._mu_coef[1] * ens_mean
            raw_sigma = self._sigma_coef[0] + self._sigma_coef[1] * ens_spread

        sigma = _softplus(raw_sigma) + 1e-6
        return NormalDist(mu=mu, sigma=sigma)

    def predict_bin_probs(
        self, features_df: pd.DataFrame, bins: list[tuple[float, float]]
    ) -> np.ndarray:
        """Predict bin probabilities from normal CDF differences."""

        dist = self.predict_distribution(features_df)
        mu = dist.mu[:, None]
        sigma = dist.sigma[:, None]

        lows = np.array([b[0] for b in bins], dtype=float)[None, :]
        highs = np.array([b[1] for b in bins], dtype=float)[None, :]

        cdf_high = norm.cdf(highs, loc=mu, scale=sigma)
        cdf_low = norm.cdf(lows, loc=mu, scale=sigma)
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
        raise ValueError("Cannot infer EMOS mode: expected x or ens_mean/ens_spread columns")

    def _assert_fitted(self) -> None:
        if self._mu_coef is None or self._sigma_coef is None or self._resolved_mode is None:
            raise RuntimeError("EMOS model is not fitted")


class BinProbabilityCalibrator:
    """Post-calibration layer for bin probabilities."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._models: list[IsotonicRegression | LogisticRegression] = []

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit one calibrator per bin."""

        if raw_probs.shape != outcomes.shape:
            raise ValueError("raw_probs and outcomes must have same shape")

        self._models = []
        for i in range(raw_probs.shape[1]):
            x = raw_probs[:, i]
            y = outcomes[:, i]
            if self.method == "isotonic":
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(x, y)
                self._models.append(model)
            else:
                model = LogisticRegression(random_state=0)
                model.fit(x.reshape(-1, 1), y)
                self._models.append(model)

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply per-bin calibration and renormalize rows."""

        if not self._models:
            raise RuntimeError("Calibrator is not fitted")

        calibrated = np.zeros_like(raw_probs, dtype=float)
        for i, model in enumerate(self._models):
            x = raw_probs[:, i]
            if isinstance(model, IsotonicRegression):
                calibrated[:, i] = model.predict(x)
            else:
                calibrated[:, i] = model.predict_proba(x.reshape(-1, 1))[:, 1]

        calibrated = np.clip(calibrated, 0.0, 1.0)
        sums = calibrated.sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        return cast(np.ndarray, calibrated / sums)


def calibrate_probability(raw_probability: float) -> MosOutput:
    """Backward-compatible helper for the smoke flow."""

    probability = min(max(raw_probability, 0.0), 1.0)
    return MosOutput(probability=probability)

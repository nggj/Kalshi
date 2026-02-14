"""Station-centric verification and reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.baselines import BaselinePredictor
from kalshi_temp_pipeline.pipeline.tasks.distributions import PredictiveDistribution


@dataclass(frozen=True)
class MethodMetrics:
    """Metrics for one forecast method."""

    mae: float
    rmse: float
    bias: float
    brier: float


def _realized_bin_index(observed_tmax: float, bins: list[tuple[float, float]]) -> int:
    for i, (low, high) in enumerate(bins):
        if low <= observed_tmax < high:
            return i
    raise ValueError("Observed value does not fall into any bin")


def _deterministic_bin_probs(pred_tmax: float, bins: list[tuple[float, float]]) -> np.ndarray:
    probs = np.zeros(len(bins), dtype=float)
    idx = _realized_bin_index(pred_tmax, bins)
    probs[idx] = 1.0
    return probs


def brier_score_multiclass(probs: np.ndarray, outcome_index: np.ndarray) -> float:
    """Compute multiclass Brier score for probs (N,K) and integer outcomes (N,)."""

    if probs.ndim != 2:
        raise ValueError("probs must be 2D with shape (N, K)")
    if outcome_index.ndim != 1:
        raise ValueError("outcome_index must be 1D with shape (N,)")
    n_samples, n_classes = probs.shape
    if outcome_index.shape[0] != n_samples:
        raise ValueError("probs and outcome_index lengths must match")

    if np.any(outcome_index < 0) or np.any(outcome_index >= n_classes):
        raise ValueError("outcome_index values must be in [0, K-1]")

    onehot = np.zeros_like(probs, dtype=float)
    onehot[np.arange(n_samples), outcome_index] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_bins_binary(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Reliability-bin summary for binary events.

    Returns dict with bin_count, confidence_mean, accuracy_mean, bin_edges.
    """

    if probs.ndim != 1 or outcomes.ndim != 1:
        raise ValueError("probs and outcomes must be 1D")
    if probs.shape[0] != outcomes.shape[0]:
        raise ValueError("probs and outcomes length must match")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=int)
    conf = np.zeros(n_bins, dtype=float)
    acc = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= left) & (probs < right)
        else:
            mask = (probs >= left) & (probs <= right)

        if np.any(mask):
            counts[i] = int(np.sum(mask))
            conf[i] = float(np.mean(probs[mask]))
            acc[i] = float(np.mean(outcomes[mask]))

    return {
        "bin_count": counts,
        "confidence_mean": conf,
        "accuracy_mean": acc,
        "bin_edges": bin_edges,
    }


def expected_calibration_error_binary(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE) for binary event predictions."""

    rel = reliability_bins_binary(probs, outcomes, n_bins=n_bins)
    counts = rel["bin_count"]
    conf = rel["confidence_mean"]
    acc = rel["accuracy_mean"]

    n = np.sum(counts)
    if n == 0:
        return 0.0

    weights = counts / n
    return float(np.sum(weights * np.abs(acc - conf)))


def _brier_score(pred_probs: np.ndarray, outcome_onehot: np.ndarray) -> float:
    return float(np.mean((pred_probs - outcome_onehot) ** 2))


def compute_metrics(
    pred_tmax: float, observed_tmax: float, pred_probs: np.ndarray, realized_idx: int
) -> MethodMetrics:
    """Compute deterministic + probabilistic metrics for one forecast."""

    err = pred_tmax - observed_tmax
    outcome = np.zeros_like(pred_probs)
    outcome[realized_idx] = 1.0
    return MethodMetrics(
        mae=abs(err),
        rmse=float(np.sqrt(err**2)),
        bias=err,
        brier=_brier_score(pred_probs, outcome),
    )


def evaluate_station(
    *,
    target_date: date,
    station: str,
    observed_tmax: float,
    raw_model_tmax: float,
    mos_dist: PredictiveDistribution,
    mos_bin_probs: np.ndarray,
    bins: list[tuple[float, float]],
    baselines: dict[str, BaselinePredictor],
) -> dict[str, MethodMetrics]:
    """Evaluate raw model, MOS, and baseline predictors for one station/date."""

    realized_idx = _realized_bin_index(observed_tmax, bins)
    metrics: dict[str, MethodMetrics] = {}

    raw_probs = _deterministic_bin_probs(raw_model_tmax, bins)
    metrics["raw_model_station"] = compute_metrics(
        raw_model_tmax, observed_tmax, raw_probs, realized_idx
    )

    mos_mean = float(mos_dist.mean()[0])
    metrics["mos_mean"] = compute_metrics(mos_mean, observed_tmax, mos_bin_probs[0], realized_idx)

    for name, predictor in baselines.items():
        pred = predictor.predict_tmax(target_date, station)
        pred_probs = _deterministic_bin_probs(pred, bins)
        metrics[name] = compute_metrics(pred, observed_tmax, pred_probs, realized_idx)

    return metrics


def generate_station_report(
    *,
    target_date: date,
    station: str,
    metrics: dict[str, MethodMetrics],
    output_root: Path = Path("artifacts/reports"),
    show_hier_bias: bool = False,
    station_bias_estimate: float | None = None,
    season_bias_estimate: float | None = None,
) -> Path:
    """Generate markdown report comparing methods."""

    output_dir = output_root / target_date.isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{station}.md"

    lines = [
        f"# Station Verification Report: {station} ({target_date.isoformat()})",
        "",
        "## Edge summary",
        "- Settlement-aligned station verification matters because Kalshi resolves",
        "  against station-relevant climate observations.",
        "- Regional model station extraction can capture local terrain/sea-breeze/urban",
        "  signals that coarse public guidance may smooth out.",
        "- MOS is required to convert that raw edge into calibrated bin",
        "  probabilities for robust EV-based decisions.",
        "",
        "## Metrics comparison",
        "| Method | MAE | RMSE | Bias | Brier |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, m in metrics.items():
        lines.append(f"| {name} | {m.mae:.3f} | {m.rmse:.3f} | {m.bias:.3f} | {m.brier:.3f} |")

    if show_hier_bias:
        station_txt = "n/a" if station_bias_estimate is None else f"{station_bias_estimate:.3f}"
        season_txt = "n/a" if season_bias_estimate is None else f"{season_bias_estimate:.3f}"
        lines.extend(
            [
                "",
                "## Hierarchical bias correction",
                f"- station_bias estimate: {station_txt}",
                f"- season_bias estimate (target date month={target_date.month}): {season_txt}",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def summarize_verification(
    bin_probs: np.ndarray, realized_bin_index: int | np.ndarray
) -> dict[str, float]:
    """Summarize core probabilistic verification metrics.

    Note: deterministic MAE/RMSE are not available from bin-only inputs.
    """

    probs_2d = np.atleast_2d(bin_probs.astype(float))

    if isinstance(realized_bin_index, int):
        outcome_idx = np.array([realized_bin_index], dtype=int)
    else:
        outcome_idx = realized_bin_index.astype(int)

    if outcome_idx.ndim != 1:
        raise ValueError("realized_bin_index must be int or 1D array")
    if outcome_idx.shape[0] != probs_2d.shape[0]:
        raise ValueError("realized_bin_index length must match number of rows in bin_probs")

    brier = brier_score_multiclass(probs_2d, outcome_idx)

    # ECE for first bin treated as binary event; useful, stable smoke metric.
    probs_bin0 = probs_2d[:, 0]
    outcomes_bin0 = (outcome_idx == 0).astype(float)
    ece_bin0 = expected_calibration_error_binary(probs_bin0, outcomes_bin0, n_bins=10)

    return {
        "brier_score": brier,
        "ece_bin0": ece_bin0,
        "mae": float("nan"),
        "rmse": float("nan"),
    }

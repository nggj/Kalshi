"""Station-centric verification and reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.baselines import BaselinePredictor
from kalshi_temp_pipeline.pipeline.tasks.mos import NormalDist


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
    mos_dist: NormalDist,
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

    mos_mean = float(mos_dist.mu[0])
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

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def summarize_verification() -> dict[str, float]:
    """Backward-compatible placeholder summary."""

    return {"mae": 0.0, "rmse": 0.0, "hit_rate": 1.0}

"""MOS/Calibration task placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MosOutput:
    """Output distribution summary for a target bin."""

    probability: float


def calibrate_probability(raw_probability: float) -> MosOutput:
    """Identity calibration placeholder."""

    probability = min(max(raw_probability, 0.0), 1.0)
    return MosOutput(probability=probability)

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.verify import (
    brier_score_multiclass,
    expected_calibration_error_binary,
    reliability_bins_binary,
    summarize_verification,
)


def test_brier_multiclass_perfect_prediction() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 2])
    assert brier_score_multiclass(probs, outcomes) == 0.0


def test_brier_multiclass_uniform_prediction() -> None:
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    outcomes = np.array([0, 1])
    # each sample sum((p-y)^2)=0.5, mean=0.5
    assert brier_score_multiclass(probs, outcomes) == 0.5


def test_reliability_and_ece_sanity() -> None:
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])

    rel = reliability_bins_binary(probs, outcomes, n_bins=2)
    assert rel["bin_count"].tolist() == [2, 2]
    np.testing.assert_allclose(rel["confidence_mean"], [0.15, 0.85], atol=1e-12)
    np.testing.assert_allclose(rel["accuracy_mean"], [0.0, 1.0], atol=1e-12)

    ece = expected_calibration_error_binary(probs, outcomes, n_bins=2)
    # 0.5*|0-0.15| + 0.5*|1-0.85| = 0.15
    np.testing.assert_allclose(ece, 0.15, atol=1e-12)


def test_summarize_verification_outputs_core_metrics() -> None:
    probs = np.array([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    realized = np.array([1, 2])
    summary = summarize_verification(probs, realized)

    assert "brier_score" in summary
    assert "ece_bin0" in summary
    assert np.isfinite(summary["brier_score"])
    assert np.isfinite(summary["ece_bin0"])

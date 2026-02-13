import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.mos import BinProbabilityCalibrator, EmosModel


def test_emos_deterministic_fit_predict_positive_sigma() -> None:
    train_df = pd.DataFrame(
        {
            "x": [70.0, 72.0, 75.0, 78.0, 80.0],
            "y": [71.0, 73.0, 74.5, 79.0, 81.0],
        }
    )
    model = EmosModel(mode="deterministic")
    model.fit(train_df)

    pred = model.predict_distribution(pd.DataFrame({"x": [74.0, 79.0]}))
    assert np.all(np.isfinite(pred.mu))
    assert np.all(np.isfinite(pred.sigma))
    assert np.all(pred.sigma > 0)


def test_emos_ensemble_bin_probs_normalized() -> None:
    train_df = pd.DataFrame(
        {
            "ens_mean": [70.0, 72.0, 75.0, 78.0, 80.0, 82.0],
            "ens_spread": [1.5, 1.6, 2.0, 2.3, 2.1, 2.5],
            "y": [69.8, 72.3, 75.8, 77.9, 80.5, 82.4],
        }
    )
    model = EmosModel(mode="ensemble")
    model.fit(train_df)

    features_df = pd.DataFrame(
        {
            "ens_mean": [76.0, 81.0],
            "ens_spread": [2.0, 2.4],
        }
    )
    bins = [(-np.inf, 75.0), (75.0, 80.0), (80.0, np.inf)]
    probs = model.predict_bin_probs(features_df, bins)

    assert probs.shape == (2, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), np.ones(2), atol=1e-6)


def test_calibrator_isotonic_monotonic_and_normalized() -> None:
    raw_probs = np.array(
        [
            [0.10, 0.90],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.40, 0.60],
            [0.50, 0.50],
        ]
    )
    outcomes = np.array(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ]
    )

    calibrator = BinProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_probs, outcomes)
    calibrated = calibrator.transform(raw_probs)

    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert np.allclose(calibrated.sum(axis=1), np.ones(raw_probs.shape[0]), atol=1e-6)

    order = np.argsort(raw_probs[:, 0])
    first_bin_sorted = calibrated[order, 0]
    assert np.all(np.diff(first_bin_sorted) >= -1e-12)

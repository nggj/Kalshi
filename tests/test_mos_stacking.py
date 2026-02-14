import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.mos_stacking import StackedGaussianMosModel


def _make_synth(n: int = 500, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    doy = (t % 365) + 1
    angle = 2.0 * np.pi * (doy / 365.25)
    doy_sin = np.sin(angle)
    doy_cos = np.cos(angle)

    true_signal = 70.0 + 8.0 * doy_sin + 3.0 * doy_cos

    # complementary errors
    e_shared = rng.normal(0.0, 0.8, size=n)
    e1 = rng.normal(0.0, 1.6, size=n)
    e2 = rng.normal(0.0, 1.6, size=n)

    wrf = true_signal + e_shared + e1
    public = true_signal + e_shared - e1
    y = true_signal + 0.7 * e_shared + 0.5 * e2

    public[rng.random(n) < 0.1] = np.nan

    return pd.DataFrame(
        {
            "wrf_tmax": wrf,
            "public_tmax": public,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "y": y,
        }
    )


def test_stacking_reduces_mae_vs_single_predictor_baselines() -> None:
    df = _make_synth()
    train = df.iloc[:350].reset_index(drop=True)
    test = df.iloc[350:].reset_index(drop=True)

    model = StackedGaussianMosModel(alpha=0.5)
    model.fit(train)

    pred = model.predict_distribution(test).mean()
    y_test = test["y"].to_numpy(dtype=float)

    wrf_mae = np.mean(np.abs(test["wrf_tmax"].to_numpy(dtype=float) - y_test))
    public_filled = test["public_tmax"].fillna(train["public_tmax"].mean()).to_numpy(dtype=float)
    public_mae = np.mean(np.abs(public_filled - y_test))
    stacked_mae = np.mean(np.abs(pred - y_test))

    assert stacked_mae < wrf_mae
    assert stacked_mae < public_mae


def test_stacking_sigma_positive_and_bin_probs_normalized() -> None:
    df = _make_synth(n=300, seed=77)
    train = df.iloc[:220].reset_index(drop=True)
    test = df.iloc[220:].reset_index(drop=True)

    model = StackedGaussianMosModel(alpha=1.0)
    model.fit(train)
    dist = model.predict_distribution(test)

    assert np.all(dist.sigma > 0.0)

    bins = [(-np.inf, 65.0), (65.0, 75.0), (75.0, np.inf)]
    probs = model.predict_bin_probs(test, bins)

    assert probs.shape == (len(test), 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), np.ones(len(test)), atol=1e-6)

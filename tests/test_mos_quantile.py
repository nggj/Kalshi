import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.mos_quantile import QuantileRegressionMosModel


def test_quantile_mos_skewed_data_asymmetric_bins() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 300)
    exp_noise = rng.exponential(scale=1.0, size=x.shape[0])
    y = x + exp_noise  # right-skewed

    train_df = pd.DataFrame({"x": x, "y": y})

    model = QuantileRegressionMosModel(mode="deterministic")
    model.fit(train_df)

    features_df = pd.DataFrame({"x": [5.0]})
    bins = [(-1e9, 5.5), (5.5, 7.0), (7.0, 1e9)]
    probs = model.predict_bin_probs(features_df, bins)

    assert probs.shape == (1, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-6)

    dist = model.predict_distribution(features_df)
    q_levels = dist.q_levels
    q_values = dist.q_values[0]
    q05 = float(q_values[np.argmin(np.abs(q_levels - 0.05))])
    q50 = float(q_values[np.argmin(np.abs(q_levels - 0.50))])
    q95 = float(q_values[np.argmin(np.abs(q_levels - 0.95))])
    # right skew: upper tail spread should exceed lower tail spread
    assert (q95 - q50) > (q50 - q05)
    assert probs[0, 0] != probs[0, 2]


def test_quantile_dist_cdf_monotone() -> None:
    rng = np.random.default_rng(42)
    ens_mean = rng.normal(loc=5.0, scale=1.0, size=250)
    ens_spread = np.clip(rng.normal(loc=1.2, scale=0.2, size=250), 0.2, None)
    y = ens_mean + rng.exponential(scale=ens_spread)

    train_df = pd.DataFrame(
        {
            "ens_mean": ens_mean,
            "ens_spread": ens_spread,
            "y": y,
        }
    )

    model = QuantileRegressionMosModel(mode="ensemble")
    model.fit(train_df)

    features_df = pd.DataFrame({"ens_mean": [5.0], "ens_spread": [1.0]})
    dist = model.predict_distribution(features_df)

    xs = np.array([3.0, 5.0, 7.0, 10.0])
    cdfs = np.array([dist.cdf(np.array([v]))[0] for v in xs])
    assert np.all(np.diff(cdfs) >= -1e-12)
    assert cdfs[0] <= cdfs[-1]

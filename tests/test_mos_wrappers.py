import numpy as np
import pandas as pd
from scipy.stats import norm

from kalshi_temp_pipeline.pipeline.tasks.mos import EmosModel
from kalshi_temp_pipeline.pipeline.tasks.mos_wrappers import (
    MixtureWrappedEmosModel,
    SkewWrappedEmosModel,
)


def test_mixture_wrapper_has_heavier_tails_than_normal() -> None:
    rng = np.random.default_rng(7)
    n = 1200
    x = rng.normal(0.0, 1.0, size=n)
    mu_true = 65.0 + 4.0 * x
    sigma_base = 2.0
    comp = rng.random(n) < 0.2
    noise = np.where(
        comp,
        rng.normal(0.0, sigma_base * 2.5, size=n),
        rng.normal(0.0, sigma_base * 0.7, size=n),
    )
    y = mu_true + noise

    train_df = pd.DataFrame({"x": x, "y": y})

    base = EmosModel(mode="deterministic")
    base.fit(train_df)
    wrapped = MixtureWrappedEmosModel(mode="deterministic")
    wrapped.fit(train_df)

    feat = pd.DataFrame({"x": np.array([0.0, 0.0, 0.0])})
    base_dist = base.predict_distribution(feat)
    mix_dist = wrapped.predict_distribution(feat)

    k = 2.0
    mu = base_dist.mu
    sig = base_dist.sigma
    normal_tail = 2.0 * (1.0 - norm.cdf(k))
    mix_tail = 1.0 - (mix_dist.cdf(mu + k * sig) - mix_dist.cdf(mu - k * sig))

    assert np.all(mix_tail > normal_tail)


def test_skew_wrapper_produces_asymmetric_probabilities() -> None:
    rng = np.random.default_rng(11)
    n = 1000
    x = rng.normal(0.0, 1.0, size=n)
    mu_true = 72.0 + 3.0 * x
    # right-skewed residuals via centered gamma
    resid = rng.gamma(shape=2.0, scale=1.5, size=n) - 3.0
    y = mu_true + resid

    train_df = pd.DataFrame({"x": x, "y": y})
    model = SkewWrappedEmosModel(mode="deterministic")
    model.fit(train_df)

    feat = pd.DataFrame({"x": [0.0]})
    dist = model.predict_distribution(feat)
    mu = dist.mean()

    left = dist.cdf(mu) - dist.cdf(mu - 1.0)
    right = dist.cdf(mu + 1.0) - dist.cdf(mu)

    assert not np.isclose(right[0], left[0])
    assert dist.cdf(mu)[0] > 0.5


def test_wrapped_bin_probs_are_valid_and_normalized() -> None:
    train_df = pd.DataFrame(
        {
            "ens_mean": [70.0, 72.0, 75.0, 78.0, 80.0, 82.0, 84.0],
            "ens_spread": [1.2, 1.4, 1.6, 2.1, 2.3, 2.5, 2.8],
            "y": [69.5, 72.4, 75.2, 77.6, 81.2, 82.7, 83.6],
        }
    )
    bins = [(-np.inf, 75.0), (75.0, 80.0), (80.0, np.inf)]
    features_df = pd.DataFrame({"ens_mean": [76.0, 81.0], "ens_spread": [2.0, 2.4]})

    for model in (MixtureWrappedEmosModel(mode="ensemble"), SkewWrappedEmosModel(mode="ensemble")):
        model.fit(train_df)
        probs = model.predict_bin_probs(features_df, bins)
        assert probs.shape == (2, 3)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), np.ones(2), atol=1e-6)

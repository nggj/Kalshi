from datetime import datetime, timedelta

import numpy as np

from kalshi_temp_pipeline.pipeline.tasks.distributions import (
    MixtureNormalDist,
    NormalDist,
    QuantileDist,
    condition_on_minimum,
)
from kalshi_temp_pipeline.pipeline.tasks.nowcast import max_so_far, nowcast_bin_probs


def test_condition_on_minimum_removes_mass_below_lower() -> None:
    dist = NormalDist(mu=np.array([80.0]), sigma=np.array([2.0]))
    conditioned = condition_on_minimum(dist, lower=82.0)

    below = conditioned.cdf(np.array([81.99]))[0]
    at = conditioned.cdf(np.array([82.0]))[0]

    assert below == 0.0
    assert np.isclose(at, 0.0, atol=1e-6)


def test_nowcast_bin_probs_normalized() -> None:
    q_levels = np.linspace(0.05, 0.95, 19)
    q_vals = np.array([[75.0 + 10.0 * q for q in q_levels]], dtype=float)
    dist = QuantileDist(q_levels=q_levels, q_values=q_vals)

    bins = [(-np.inf, 80.0), (80.0, 85.0), (85.0, np.inf)]
    probs = nowcast_bin_probs(dist, bins, max_so_far_value=83.0)

    assert probs.shape == (1, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), np.ones(1), atol=1e-6)
    assert probs[0, 0] <= 1e-6


def test_very_low_lower_bound_keeps_distribution_effectively_unchanged() -> None:
    dist = MixtureNormalDist(
        weights=np.array([[0.7, 0.3]]),
        mus=np.array([[82.0, 84.0]]),
        sigmas=np.array([[1.5, 2.0]]),
    )
    conditioned = condition_on_minimum(dist, lower=-1e9)

    x = np.array([80.0])
    assert np.isclose(conditioned.cdf(x)[0], dist.cdf(x)[0], atol=1e-8)


def test_max_so_far_uses_as_of_filter() -> None:
    t0 = datetime(2026, 2, 13, 9, 0)
    times = [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)]
    temps = [75.0, 79.5, 78.0]

    assert max_so_far(times, temps, t0 + timedelta(hours=1, minutes=30)) == 79.5

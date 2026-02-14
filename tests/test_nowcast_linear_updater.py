import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.distributions import NormalDist
from kalshi_temp_pipeline.pipeline.tasks.nowcast import NowcastLinearUpdater


def test_near_close_shrink_smaller_than_far() -> None:
    rng = np.random.default_rng(1)
    n = 400
    lead = rng.uniform(0.0, 24.0, size=n)
    mu_base = np.full(n, 80.0)
    sigma_base = np.full(n, 3.0)
    now_err = rng.normal(0.0, 2.0, size=n)

    near = lead <= 6.0
    noise = np.where(near, rng.normal(0.0, 0.4, size=n), rng.normal(0.0, 1.2, size=n))
    y = mu_base + np.where(near, 0.8, 0.2) * now_err + noise

    df = pd.DataFrame(
        {
            "lead_hours": lead,
            "mu_base": mu_base,
            "sigma_base": sigma_base,
            "temp_obs_now": 70.0 + now_err,
            "temp_fcst_now": np.full(n, 70.0),
            "y": y,
        }
    )

    upd = NowcastLinearUpdater(near_cutoff_hours=6.0)
    upd.fit(df)

    assert upd.shrink_by_bucket["near"] < upd.shrink_by_bucket["far"]


def test_positive_k_and_reduces_error() -> None:
    rng = np.random.default_rng(2)
    n = 300
    lead = rng.uniform(0.0, 24.0, size=n)
    mu_base = np.full(n, 75.0)
    sigma_base = np.full(n, 2.0)
    now_err = rng.normal(0.0, 1.5, size=n)

    y = mu_base + 0.7 * now_err + rng.normal(0.0, 0.3, size=n)
    df = pd.DataFrame(
        {
            "lead_hours": lead,
            "mu_base": mu_base,
            "sigma_base": sigma_base,
            "temp_obs_now": 65.0 + now_err,
            "temp_fcst_now": np.full(n, 65.0),
            "y": y,
        }
    )

    upd = NowcastLinearUpdater()
    upd.fit(df)

    assert upd.k_by_bucket["near"] > 0.0
    assert upd.k_by_bucket["far"] > 0.0

    base = NormalDist(mu=mu_base.copy(), sigma=sigma_base.copy())
    adj = upd.update_distribution(
        base,
        lead_hours=lead,
        temp_obs_now=df["temp_obs_now"].to_numpy(dtype=float),
        temp_fcst_now=df["temp_fcst_now"].to_numpy(dtype=float),
    )

    base_mae = float(np.mean(np.abs(mu_base - y)))
    adj_mae = float(np.mean(np.abs(adj.mu - y)))
    assert adj_mae < base_mae

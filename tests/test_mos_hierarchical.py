import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.hierarchical import HierarchicalBiasCorrector
from kalshi_temp_pipeline.pipeline.tasks.mos import EmosModel
from kalshi_temp_pipeline.pipeline.tasks.mos_hierarchical import HierarchicalCorrectedMosModel
from kalshi_temp_pipeline.pipeline.tasks.mos_quantile import QuantileRegressionMosModel


def _make_station_data(
    n_per_station: int = 120, seed: int = 5, season_amp: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stations = ["KNYC", "KSEA", "KORD"]
    station_bias = {"KNYC": 2.0, "KSEA": -1.8, "KORD": 0.7}

    rows: list[dict[str, float | str]] = []
    for i, stn in enumerate(stations):
        x = rng.normal(loc=0.0, scale=1.0, size=n_per_station)
        for j in range(n_per_station):
            month = ((j % 12) + 1)
            season = season_amp if month <= 6 else -season_amp
            y = 70.0 + 4.0 * x[j] + station_bias[stn] + season + rng.normal(0.0, 1.0)
            rows.append(
                {
                    "station": stn,
                    "date": f"2026-{month:02d}-{(j % 28) + 1:02d}",
                    "x": float(x[j]),
                    "y": float(y),
                }
            )

    out = pd.DataFrame(rows)
    return out.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_hierarchical_wrapper_improves_mae_for_normal_model() -> None:
    df = _make_station_data(season_amp=0.0)
    train = df.iloc[:260].reset_index(drop=True)
    test = df.iloc[260:].reset_index(drop=True)

    base = EmosModel(mode="deterministic")
    base.fit(train)
    base_pred = base.predict_distribution(test).mean()

    wrapped = HierarchicalCorrectedMosModel(
        base_model=EmosModel(mode="deterministic"),
        bias_corrector=HierarchicalBiasCorrector(),
    )
    wrapped.fit(train)
    wrapped_pred = wrapped.predict_distribution(test).mean()

    y_test = test["y"].to_numpy(dtype=float)
    base_mae = float(np.mean(np.abs(base_pred - y_test)))
    wrapped_mae = float(np.mean(np.abs(wrapped_pred - y_test)))

    assert wrapped_mae < base_mae


def test_hierarchical_wrapper_works_with_quantile_model_and_bin_probs() -> None:
    df = _make_station_data(seed=11, season_amp=0.4)
    train = df.iloc[:250].reset_index(drop=True)
    test = df.iloc[250:].reset_index(drop=True)

    wrapped = HierarchicalCorrectedMosModel(
        base_model=QuantileRegressionMosModel(mode="deterministic"),
        bias_corrector=HierarchicalBiasCorrector(),
    )
    wrapped.fit(train)
    dist = wrapped.predict_distribution(test)

    assert dist.mean().shape[0] == len(test)

    bins = [(-np.inf, 68.0), (68.0, 74.0), (74.0, np.inf)]
    probs = wrapped.predict_bin_probs(test, bins)

    assert probs.shape == (len(test), 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), np.ones(len(test)), atol=1e-6)

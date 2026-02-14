import numpy as np
import pandas as pd

from kalshi_temp_pipeline.pipeline.tasks.hierarchical import HierarchicalBiasCorrector


def _make_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def gen(station: str, n: int, station_bias: float) -> pd.DataFrame:
        months = rng.integers(1, 13, size=n)
        # month effect in residual space
        month_bias = np.where(months <= 6, 1.0, -1.0)
        resid = station_bias + month_bias + rng.normal(0.0, 0.7, size=n)
        dates = [f"2026-{m:02d}-15" for m in months]
        return pd.DataFrame({"station": station, "date": dates, "residual": resid})

    # many samples for KNYC, few for KSEA
    return pd.concat(
        [
            gen("KNYC", 220, station_bias=2.0),
            gen("KSEA", 12, station_bias=2.0),
            gen("KORD", 140, station_bias=-1.5),
        ],
        ignore_index=True,
    )


def test_station_shrinkage_few_samples_more_than_many() -> None:
    df = _make_data()
    corrector = HierarchicalBiasCorrector()
    corrector.fit(df)

    knyc_raw = float(df.loc[df["station"] == "KNYC", "residual"].mean())
    ksea_raw = float(df.loc[df["station"] == "KSEA", "residual"].mean())

    knyc_post = corrector.station_bias["KNYC"]
    ksea_post = corrector.station_bias["KSEA"]

    # station with fewer samples should be pulled toward zero more strongly
    assert abs(ksea_post) < abs(ksea_raw)
    assert abs(knyc_post) < abs(knyc_raw)
    assert abs(ksea_post / ksea_raw) < abs(knyc_post / knyc_raw)


def test_apply_reduces_bias_out_of_sample() -> None:
    df = _make_data(seed=99)
    train = df.iloc[:280].reset_index(drop=True)
    test = df.iloc[280:].reset_index(drop=True)

    corrector = HierarchicalBiasCorrector()
    corrector.fit(train)

    # base forecast mu=0, true y=residual (bias present)
    mu_base = np.zeros(len(test), dtype=float)
    y_true = test["residual"].to_numpy(dtype=float)

    mu_adj = corrector.apply(mu=mu_base, station=test["station"].to_numpy(), date=test["date"])

    base_bias = float(np.mean(y_true - mu_base))
    adj_bias = float(np.mean(y_true - mu_adj))

    assert abs(adj_bias) < abs(base_bias)

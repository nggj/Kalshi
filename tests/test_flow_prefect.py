from kalshi_temp_pipeline.pipeline import flow as flow_module


def test_smoke_prefect_flow_returns_ok(monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")

    def _fake_fetch_daily_cli(self, station: str, year: int):
        return {"station": station, "year": year, "results": []}

    monkeypatch.setattr(
        flow_module.ObsCliClient,
        "fetch_daily_cli",
        _fake_fetch_daily_cli,
    )

    assert flow_module.smoke_prefect_flow() == "OK"

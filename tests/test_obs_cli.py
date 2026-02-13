import json
from datetime import date
from pathlib import Path

import httpx

from kalshi_temp_pipeline.pipeline.tasks.obs_cli import (
    extract_daily_tmax,
    fetch_cli_year,
)


def _load_fixture(name: str) -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8"))


def test_extract_daily_tmax_parses_required_fields() -> None:
    payload = _load_fixture("iem_cli_sample.json")
    parsed = extract_daily_tmax(payload)
    assert parsed == {date(2026, 7, 14): 89.0, date(2026, 7, 15): 91.0}


def test_fetch_cli_year_uses_cache_when_present(tmp_path: Path) -> None:
    cache_dir = tmp_path / "obs_cache"
    cache_dir.mkdir(parents=True)
    cached_payload = _load_fixture("iem_cli_sample.json")
    (cache_dir / "KNYC_2026.json").write_text(json.dumps(cached_payload), encoding="utf-8")

    class NeverCalledTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("Network should not be called when cache exists")

    with httpx.Client(transport=NeverCalledTransport()) as client:
        got = fetch_cli_year("KNYC", 2026, cache_dir=cache_dir, client=client)

    assert got == cached_payload


def test_fetch_cli_year_writes_cache_after_http(tmp_path: Path) -> None:
    cache_dir = tmp_path / "obs_cache"
    payload = _load_fixture("iem_cli_sample.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/json/cli.py"
        assert request.url.params["station"] == "KNYC"
        assert request.url.params["year"] == "2026"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        got = fetch_cli_year("KNYC", 2026, cache_dir=cache_dir, client=client)

    assert got == payload
    cache_file = cache_dir / "KNYC_2026.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == payload

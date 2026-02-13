import json
from pathlib import Path

import httpx
import pytest

from kalshi_temp_pipeline.pipeline.tasks.kalshi_api import KalshiClient


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_get_methods_with_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("KALSHI_API_KEY", "k")
    monkeypatch.setenv("KALSHI_API_SECRET", "s")

    market_payload = _fixture("kalshi_market_sample.json")
    orderbook_payload = _fixture("kalshi_orderbook_sample.json")
    candles_payload = _fixture("kalshi_candles_sample.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trade-api/v2/markets"):
            return httpx.Response(200, json=market_payload)
        if request.url.path.endswith("/trade-api/v2/markets/orderbook"):
            return httpx.Response(200, json=orderbook_payload)
        if request.url.path.endswith("/trade-api/v2/series/candlesticks"):
            return httpx.Response(200, json=candles_payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = KalshiClient(base_url="https://example.test", client=client)
        assert api.get_market("KXHIGHTEMP-DSM") == market_payload
        assert api.get_orderbook("KXHIGHTEMP-DSM") == orderbook_payload
        assert api.get_candlesticks("KXHIGHTEMP", 60) == candles_payload


def test_place_order_dry_run_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")

    class NoPostTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("No HTTP requests expected in DRY_RUN place_order")

    with httpx.Client(transport=NoPostTransport()) as client:
        api = KalshiClient(base_url="https://example.test", client=client)
        response = api.place_order(
            ticker="KXHIGHTEMP-DSM",
            side="yes",
            action="buy",
            count=1,
            yes_price_cents=61,
            client_order_id="abc-123",
        )

    assert response["status"] == "dry_run"
    assert response["sent"] is False
    assert response["payload"]["ticker"] == "KXHIGHTEMP-DSM"


def test_place_order_live_mode_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRY_RUN", "false")
    placed_payload = _fixture("kalshi_order_placed_sample.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/trade-api/v2/portfolio/orders")
        return httpx.Response(200, json=placed_payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = KalshiClient(base_url="https://example.test", client=client)
        response = api.place_order(
            ticker="KXHIGHTEMP-DSM",
            side="yes",
            action="buy",
            count=1,
            yes_price_cents=61,
            client_order_id="abc-123",
        )

    assert response == placed_payload

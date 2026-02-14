import base64
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_temp_pipeline.pipeline.tasks.kalshi_api import KalshiClient


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _write_temp_private_key(tmp_path: Path) -> tuple[Path, rsa.RSAPublicKey]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "kalshi_test_private_key.pem"
    key_path.write_bytes(pem)
    return key_path, public_key


def test_get_methods_with_mocked_http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "key-id-123")
    key_path, public_key = _write_temp_private_key(tmp_path)
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    fixed_ts = 1700000000123

    market_payload = _fixture("kalshi_market_sample.json")
    orderbook_payload = _fixture("kalshi_orderbook_sample.json")
    candles_payload = _fixture("kalshi_candles_sample.json")

    def _assert_auth_headers(request: httpx.Request) -> None:
        assert request.headers["KALSHI-ACCESS-KEY"] == "key-id-123"
        assert request.headers["KALSHI-ACCESS-TIMESTAMP"] == str(fixed_ts)
        signature_b64 = request.headers["KALSHI-ACCESS-SIGNATURE"]
        signature = base64.b64decode(signature_b64)
        message = f"{fixed_ts}{request.method}{request.url.path}".encode("utf-8")
        public_key.verify(
            signature,
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )

    def handler(request: httpx.Request) -> httpx.Response:
        _assert_auth_headers(request)

        if request.url.path.endswith("/trade-api/v2/markets"):
            return httpx.Response(200, json=market_payload)
        if request.url.path.endswith("/trade-api/v2/markets/orderbook"):
            return httpx.Response(200, json=orderbook_payload)
        if request.url.path.endswith("/trade-api/v2/series/candlesticks"):
            return httpx.Response(200, json=candles_payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = KalshiClient(
            base_url="https://example.test", client=client, clock_ms=lambda: fixed_ts
        )
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


def test_place_order_live_mode_sends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "key-id-live")
    key_path, _ = _write_temp_private_key(tmp_path)
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    placed_payload = _fixture("kalshi_order_placed_sample.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/trade-api/v2/portfolio/orders")
        assert "KALSHI-ACCESS-SIGNATURE" in request.headers
        return httpx.Response(200, json=placed_payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = KalshiClient(
            base_url="https://example.test",
            client=client,
            clock_ms=lambda: 1700000000000,
        )
        response = api.place_order(
            ticker="KXHIGHTEMP-DSM",
            side="yes",
            action="buy",
            count=1,
            yes_price_cents=61,
            client_order_id="abc-123",
        )

    assert response == placed_payload

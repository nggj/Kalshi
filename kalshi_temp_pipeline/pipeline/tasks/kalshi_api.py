"""Kalshi API wrapper with DRY_RUN safety and official signed auth headers."""

from __future__ import annotations

import base64
import logging
from os import getenv
from pathlib import Path
from time import time
from typing import Any, Callable, cast

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)


class KalshiClient:
    """HTTP client wrapper for Kalshi endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        env_base = getenv("KALSHI_API_BASE_URL")
        self.base_url = base_url or env_base or "https://api.elections.kalshi.com"
        self.api_key_id = getenv("KALSHI_API_KEY_ID", "")
        self.private_key_pem_path = getenv("KALSHI_PRIVATE_KEY_PEM_PATH", "")
        self.dry_run = getenv("DRY_RUN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.timeout_s = timeout_s
        self._client = client
        self._clock_ms = clock_ms or (lambda: int(time() * 1000))
        self._private_key = self._load_private_key(self.private_key_pem_path)

    def _load_private_key(self, pem_path: str) -> rsa.RSAPrivateKey | None:
        if not pem_path:
            return None
        pem_bytes = Path(pem_path).read_bytes()
        key = serialization.load_pem_private_key(pem_bytes, password=None)
        return cast(rsa.RSAPrivateKey, key)

    def _signed_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is required for authenticated requests")
        if self._private_key is None:
            raise ValueError("KALSHI_PRIVATE_KEY_PEM_PATH is required for authenticated requests")

        ts = str(self._clock_ms())
        path_no_query = path.split("?", maxsplit=1)[0]
        message = f"{ts}{method.upper()}{path_no_query}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode("utf-8")
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
        }

    def _request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = self._signed_headers(method, path)
        if self._client is not None:
            response = self._client.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

        with httpx.Client(timeout=self.timeout_s) as owned_client:
            response = owned_client.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = self._signed_headers("POST", path)
        if self._client is not None:
            response = self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

        with httpx.Client(timeout=self.timeout_s) as owned_client:
            response = owned_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    def get_market(self, ticker: str) -> dict[str, Any]:
        """Fetch market metadata."""

        return self._request("GET", "/trade-api/v2/markets", params={"ticker": ticker})

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch market orderbook."""

        return self._request("GET", "/trade-api/v2/markets/orderbook", params={"ticker": ticker})

    def get_candlesticks(
        self,
        series_ticker: str,
        period_interval: int,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> dict[str, Any]:
        """Fetch candlestick history for a series ticker."""

        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "period_interval": period_interval,
        }
        if start_ts is not None:
            params["start_ts"] = start_ts
        if end_ts is not None:
            params["end_ts"] = end_ts

        return self._request("GET", "/trade-api/v2/series/candlesticks", params=params)

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        count: int,
        yes_price_cents: int,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Place order, or return dry-run stub when DRY_RUN is enabled."""

        payload = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "yes_price_cents": yes_price_cents,
            "client_order_id": client_order_id,
        }

        if self.dry_run:
            logger.info("DRY_RUN enabled: order intent %s", payload)
            return {
                "status": "dry_run",
                "sent": False,
                "payload": payload,
            }

        return self._post("/trade-api/v2/portfolio/orders", payload)

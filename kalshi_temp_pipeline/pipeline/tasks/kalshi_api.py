"""Kalshi API wrapper with DRY_RUN safety and env-based credentials."""

from __future__ import annotations

import logging
from os import getenv
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)


class KalshiClient:
    """HTTP client wrapper for Kalshi endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        env_base = getenv("KALSHI_API_BASE_URL")
        self.base_url = base_url or env_base or "https://api.elections.kalshi.com"
        self.api_key = getenv("KALSHI_API_KEY", "")
        self.api_secret = getenv("KALSHI_API_SECRET", "")
        self.dry_run = getenv("DRY_RUN", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.timeout_s = timeout_s
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["KALSHI-ACCESS-KEY"] = self.api_key
        if self.api_secret:
            headers["KALSHI-ACCESS-SECRET"] = self.api_secret
        return headers

    def _request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        if self._client is not None:
            response = self._client.request(method, url, headers=self._headers(), params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

        with httpx.Client(timeout=self.timeout_s) as owned_client:
            response = owned_client.request(method, url, headers=self._headers(), params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        if self._client is not None:
            response = self._client.post(url, headers=self._headers(), json=body)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

        with httpx.Client(timeout=self.timeout_s) as owned_client:
            response = owned_client.post(url, headers=self._headers(), json=body)
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

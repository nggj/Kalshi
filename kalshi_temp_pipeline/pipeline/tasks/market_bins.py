"""Temperature market-bin parsing utilities."""

from __future__ import annotations

import math
import re


def _half_integer_bounds(low_int: int | None, high_int: int | None) -> tuple[float, float]:
    low = -math.inf if low_int is None else float(low_int) - 0.5
    high = math.inf if high_int is None else float(high_int) + 0.5
    return low, high


def _as_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError("strike value is not int-convertible")


def _parse_from_subtitle(subtitle: str) -> tuple[float, float]:
    text = subtitle.strip().lower()

    range_match = re.search(r"(\d+)\s*°\s*to\s*(\d+)\s*°", text)
    if range_match:
        low_i = int(range_match.group(1))
        high_i = int(range_match.group(2))
        return _half_integer_bounds(low_i, high_i)

    below_match = re.search(r"(\d+)\s*°\s*or\s*below", text)
    if below_match:
        cap_i = int(below_match.group(1))
        return _half_integer_bounds(None, cap_i)

    above_match = re.search(r"(\d+)\s*°\s*or\s*above", text)
    if above_match:
        floor_i = int(above_match.group(1))
        return _half_integer_bounds(floor_i, None)

    raise ValueError(f"Could not parse temperature bin subtitle: {subtitle}")


def parse_temp_bin_from_market(market: dict[str, object]) -> tuple[float, float]:
    """Parse temperature bin bounds [low, high) from market metadata."""

    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    strike_type = str(market.get("strike_type", "")).lower()

    # Preferred path: structured fields
    if strike_type and (floor is not None or cap is not None):
        low_i = _as_int(floor)
        high_i = _as_int(cap)

        if strike_type == "between":
            if low_i is None or high_i is None:
                raise ValueError("between strike_type requires both floor_strike and cap_strike")
            return _half_integer_bounds(low_i, high_i)
        if strike_type in {"lte", "le", "below"}:
            if high_i is None:
                high_i = low_i
            return _half_integer_bounds(None, high_i)
        if strike_type in {"gte", "ge", "above"}:
            if low_i is None:
                low_i = high_i
            return _half_integer_bounds(low_i, None)

    # Fallback path: subtitle parsing
    subtitle = str(market.get("subtitle", ""))
    if subtitle:
        return _parse_from_subtitle(subtitle)

    raise ValueError("Market has neither parseable structured strikes nor subtitle")


def build_bins_from_markets(markets: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    """Build ticker->bin mapping from market metadata list."""

    bins: dict[str, tuple[float, float]] = {}
    for market in markets:
        ticker = str(market.get("ticker", ""))
        if not ticker:
            raise ValueError("Each market must include a ticker")
        bins[ticker] = parse_temp_bin_from_market(market)
    return bins

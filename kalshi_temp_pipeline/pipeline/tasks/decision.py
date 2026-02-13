"""Decision and EV utilities for market bins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionResult:
    """Result for a single bin decision."""

    enter: bool
    expected_value: float
    edge: float


def implied_probability(price_cents: float) -> float:
    """Convert 0-100c contract price to implied probability."""

    if price_cents < 0 or price_cents > 100:
        raise ValueError("price_cents must be between 0 and 100")
    return price_cents / 100.0


def expected_value_yes(probability: float, price_cents: float, fee_cents: float = 0.0) -> float:
    """Simple YES-side EV in cents."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    win_payout = 100.0 - price_cents - fee_cents
    lose_payout = -price_cents - fee_cents
    return probability * win_payout + (1.0 - probability) * lose_payout


def risk_allowed(
    current_exposure_usd: float, order_size_usd: float, max_exposure_usd: float
) -> bool:
    """Check basic max exposure guardrail."""

    if min(current_exposure_usd, order_size_usd, max_exposure_usd) < 0:
        raise ValueError("Exposure values must be non-negative")
    return current_exposure_usd + order_size_usd <= max_exposure_usd


def make_decision(
    probability: float,
    price_cents: float,
    edge_threshold: float,
    min_hit_rate: float,
    current_exposure_usd: float,
    order_size_usd: float,
    max_exposure_usd: float,
) -> DecisionResult:
    """Apply edge/hit-rate/risk filters for DRY_RUN decisioning."""

    implied = implied_probability(price_cents)
    edge = probability - implied
    ev = expected_value_yes(probability=probability, price_cents=price_cents)
    enter = (
        edge >= edge_threshold
        and probability >= min_hit_rate
        and risk_allowed(current_exposure_usd, order_size_usd, max_exposure_usd)
    )
    return DecisionResult(enter=enter, expected_value=ev, edge=edge)

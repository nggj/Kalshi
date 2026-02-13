"""Decision engine and EV utilities for market bins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionResult:
    """Backward-compatible single-bin decision result."""

    enter: bool
    expected_value: float
    edge: float


@dataclass(frozen=True)
class ProposedOrder:
    """Structured output for a candidate order."""

    bin_id: str
    side: str
    price_cents: float
    probability: float
    implied_probability: float
    edge: float
    expected_value_cents: float
    size_usd: float
    reason: str


def implied_probability(price_cents: float) -> float:
    """Convert 0-100c contract price to implied probability."""

    if price_cents < 0 or price_cents > 100:
        raise ValueError("price_cents must be between 0 and 100")
    return price_cents / 100.0


def expected_value_yes(
    probability: float,
    price_cents: float,
    fee_cents: float = 0.0,
    slippage_cents: float = 0.0,
) -> float:
    """Simple YES-side EV in cents including fee/slippage hooks."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    total_cost = price_cents + fee_cents + slippage_cents
    win_payout = 100.0 - total_cost
    lose_payout = -total_cost
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


def _has_sufficient_liquidity(
    row: dict[str, float], min_depth: float, max_spread_cents: float
) -> bool:
    yes_bid = float(row["yes_bid_cents"])
    yes_ask = float(row["yes_ask_cents"])
    bid_size = float(row["yes_bid_size"])
    ask_size = float(row["yes_ask_size"])

    spread = yes_ask - yes_bid
    depth = min(bid_size, ask_size)
    return spread <= max_spread_cents and depth >= min_depth


def propose_orders(
    bin_probabilities: dict[str, float],
    orderbook_snapshot: dict[str, dict[str, float]],
    *,
    edge_threshold: float,
    min_probability: float,
    fee_cents: float,
    slippage_cents: float,
    max_daily_exposure_usd: float,
    current_daily_exposure_usd: float,
    order_size_usd: float,
    min_depth_contracts: float,
    max_spread_cents: float,
) -> list[ProposedOrder]:
    """Generate deterministic order proposals from probabilities + orderbook.

    No network/API calls are made in this logic.
    """

    if max_daily_exposure_usd < 0 or current_daily_exposure_usd < 0 or order_size_usd <= 0:
        raise ValueError("Exposure and order size constraints are invalid")

    candidates: list[ProposedOrder] = []
    for bin_id, probability in bin_probabilities.items():
        if not 0.0 <= probability <= 1.0:
            raise ValueError("bin probability must be between 0 and 1")

        row = orderbook_snapshot.get(bin_id)
        if row is None:
            continue

        if not _has_sufficient_liquidity(row, min_depth_contracts, max_spread_cents):
            continue

        price = float(row["yes_ask_cents"])
        implied = implied_probability(price)
        edge = probability - implied
        if probability < min_probability or edge < edge_threshold:
            continue

        ev = expected_value_yes(
            probability=probability,
            price_cents=price,
            fee_cents=fee_cents,
            slippage_cents=slippage_cents,
        )
        if ev <= 0:
            continue

        candidates.append(
            ProposedOrder(
                bin_id=bin_id,
                side="yes",
                price_cents=price,
                probability=probability,
                implied_probability=implied,
                edge=edge,
                expected_value_cents=ev,
                size_usd=order_size_usd,
                reason="edge+liquidity+risk_pass",
            )
        )

    candidates.sort(key=lambda c: (c.expected_value_cents, c.edge), reverse=True)

    selected: list[ProposedOrder] = []
    exposure = current_daily_exposure_usd
    for candidate in candidates:
        if risk_allowed(exposure, candidate.size_usd, max_daily_exposure_usd):
            selected.append(candidate)
            exposure += candidate.size_usd

    return selected

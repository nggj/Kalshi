import json
from pathlib import Path

import pytest

from kalshi_temp_pipeline.pipeline.tasks.decision import (
    expected_value_yes,
    implied_probability,
    make_decision,
    propose_orders,
    risk_allowed,
)


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_implied_probability() -> None:
    assert implied_probability(61.0) == 0.61


def test_implied_probability_invalid() -> None:
    with pytest.raises(ValueError):
        implied_probability(101.0)


def test_expected_value_yes() -> None:
    ev = expected_value_yes(probability=0.7, price_cents=60.0, fee_cents=0.0)
    assert ev == pytest.approx(10.0)


def test_risk_allowed() -> None:
    assert risk_allowed(current_exposure_usd=20.0, order_size_usd=10.0, max_exposure_usd=40.0)
    assert not risk_allowed(current_exposure_usd=35.0, order_size_usd=10.0, max_exposure_usd=40.0)


def test_make_decision_enter_true() -> None:
    result = make_decision(
        probability=0.7,
        price_cents=60.0,
        edge_threshold=0.05,
        min_hit_rate=0.7,
        current_exposure_usd=0.0,
        order_size_usd=10.0,
        max_exposure_usd=100.0,
    )
    assert result.enter


def test_propose_orders_applies_liquidity_spread_and_risk() -> None:
    probs = {"bin_hot": 0.72, "bin_mild": 0.60, "bin_cool": 0.40}
    orderbook = _fixture("decision_orderbook_sample.json")

    orders = propose_orders(
        probs,
        orderbook,
        edge_threshold=0.05,
        min_probability=0.55,
        fee_cents=1.0,
        slippage_cents=1.0,
        max_daily_exposure_usd=20.0,
        current_daily_exposure_usd=10.0,
        order_size_usd=10.0,
        min_depth_contracts=10.0,
        max_spread_cents=3.0,
    )

    assert len(orders) == 1
    assert orders[0].bin_id == "bin_hot"
    assert orders[0].side == "yes"


def test_propose_orders_no_api_calls_pure_function_behavior() -> None:
    probs = {"bin_hot": 0.72}
    orderbook = _fixture("decision_orderbook_sample.json")

    first = propose_orders(
        probs,
        orderbook,
        edge_threshold=0.01,
        min_probability=0.50,
        fee_cents=0.0,
        slippage_cents=0.0,
        max_daily_exposure_usd=100.0,
        current_daily_exposure_usd=0.0,
        order_size_usd=10.0,
        min_depth_contracts=1.0,
        max_spread_cents=10.0,
    )
    second = propose_orders(
        probs,
        orderbook,
        edge_threshold=0.01,
        min_probability=0.50,
        fee_cents=0.0,
        slippage_cents=0.0,
        max_daily_exposure_usd=100.0,
        current_daily_exposure_usd=0.0,
        order_size_usd=10.0,
        min_depth_contracts=1.0,
        max_spread_cents=10.0,
    )

    assert first == second

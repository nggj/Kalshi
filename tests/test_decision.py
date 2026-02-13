import pytest

from kalshi_temp_pipeline.pipeline.tasks.decision import (
    expected_value_yes,
    implied_probability,
    make_decision,
    risk_allowed,
)


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

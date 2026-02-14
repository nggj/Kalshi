import math

from kalshi_temp_pipeline.pipeline.tasks.market_bins import (
    build_bins_from_markets,
    parse_temp_bin_from_market,
)


def _sample_markets() -> list[dict]:
    return [
        {"ticker": "T-LOW", "strike_type": "lte", "cap_strike": 36},
        {"ticker": "T-37-38", "strike_type": "between", "floor_strike": 37, "cap_strike": 38},
        {"ticker": "T-39-40", "subtitle": "39° to 40°"},
        {"ticker": "T-HIGH", "subtitle": "41° or above"},
    ]


def test_parse_temp_bin_from_market_structured_and_subtitle() -> None:
    market = {"strike_type": "between", "floor_strike": 37, "cap_strike": 38}
    assert parse_temp_bin_from_market(market) == (36.5, 38.5)
    assert parse_temp_bin_from_market({"subtitle": "36° or below"}) == (-math.inf, 36.5)
    assert parse_temp_bin_from_market({"subtitle": "41° or above"}) == (40.5, math.inf)


def test_build_bins_from_markets_and_contiguity() -> None:
    bins = build_bins_from_markets(_sample_markets())
    assert bins["T-LOW"] == (-math.inf, 36.5)
    assert bins["T-37-38"] == (36.5, 38.5)
    assert bins["T-39-40"] == (38.5, 40.5)
    assert bins["T-HIGH"] == (40.5, math.inf)

    ordered = sorted(bins.items(), key=lambda kv: kv[1][0])
    for i in range(len(ordered) - 1):
        _, (_, hi) = ordered[i]
        _, (next_lo, _) = ordered[i + 1]
        if math.isfinite(hi) and math.isfinite(next_lo):
            assert hi == next_lo

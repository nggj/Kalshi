-- Minimal metadata schema for MVP scaffold

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    run_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT NOT NULL,
    dry_run BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES pipeline_runs(id),
    market_ticker TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    yes_price_cents DOUBLE PRECISION NOT NULL,
    expected_value_cents DOUBLE PRECISION NOT NULL,
    entered BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

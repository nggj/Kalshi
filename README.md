# Kalshi Temp Pipeline (MVP Scaffold)

## Install
```bash
poetry install
```

## DRY_RUN pipeline smoke test
Default is `DRY_RUN=true` from `.env.example` and runtime config.

```bash
cp .env.example .env
poetry run python -m kalshi_temp_pipeline
# or
poetry run kalshi-smoke
```

Expected output includes final `OK`.


## IEM CLI station/year usage
Use `station` + `year` to fetch CLI climate JSON via `fetch_cli_year(station, year)`.
Responses are cached at `artifacts/obs_cache/{STATION}_{YEAR}.json` and reused if present.

## Tests / Lint / Typecheck
```bash
poetry run pytest -q
poetry run ruff check .
poetry run mypy .
```

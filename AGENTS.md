## Project goal
Build an MVP for a Kalshi daily high temperature pipeline with:
- strict climate-day (LST/DST) Tmax computation aligned to NWS climate reporting rules
- modular tasks (data ingest, postprocess, MOS/calibration, decision engine)
- safe execution defaults (DRY_RUN / paper trading)

## Working agreements
- Default to DRY_RUN=true. Never place real orders unless explicitly enabled.
- Do not log secrets or API keys. Read secrets from env only.
- Prefer small, reviewable commits. Keep changes minimal and testable.
- If you add a dependency, update pyproject.toml and keep it minimal.

## Tech stack
- Python 3.11+
- Poetry for deps and scripts
- Prefect for orchestration
- pytest for tests
- ruff + mypy for lint/type

## Commands
- Install: `poetry install`
- Lint: `poetry run ruff check .`
- Typecheck: `poetry run mypy .`
- Tests: `poetry run pytest -q`

## Test policy
- Any new “core logic” must have unit tests:
  - climate-day window
  - Tmax computation over the window
  - implied probability / EV calculator
  - risk guardrails

## External services policy
- Network calls must be isolated behind clients and mockable in tests.
- Provide fixtures for IEM CLI sample responses and Kalshi API sample responses.

Codex는 작업 시작 전에 AGENTS.md를 읽고 그 지침을 따른다. AGENTS.md 파일을 위의 텍스트대로 생성해줘

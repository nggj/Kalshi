# Kalshi Temp Pipeline (MVP Scaffold)

## Install
```bash
poetry install
```

## MOS/Calibration first (핵심)
이 파이프라인의 중심은 MOS/Calibration 입니다.
- 입력: station 추출 WRF Tmax + 보조 피처
- 출력: 정규 분포(\`mu\`, \`sigma\`)와 bin 확률 \`P(Tmax in bin)\`
- 목표: 점예보 정확도보다 **calibrated probability** 품질

자세한 내용은 `docs/mos.md`를 참고하세요.

## Station-centric evaluation + baseline comparison
평가는 항상 정산 기준 관측소(station) 중심으로 수행합니다.
비교 대상:
- baseline_persistence
- baseline_climatology
- baseline_public_csv (옵션, `artifacts/baselines/*.csv`)
- raw_model_station (WRF station extraction)
- mos_mean / mos calibrated probs

리포트 생성 예시(DRY_RUN, 무네트워크):
```bash
poetry run python -m kalshi_temp_pipeline.report --date 2026-02-13 --station KNYC
```

출력: `artifacts/reports/{date}/{station}.md`

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

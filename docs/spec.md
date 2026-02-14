# Kalshi 기온(일 최고기온) 자동매매 시스템 아키텍처 (운영 관점)

## 1) 전체 시스템 아키텍처

```text
[Scheduler/Orchestrator]
   ├─ (A) Kalshi 마켓 스캐너 (오늘/내일 이벤트, close time 확인)
   ├─ (B) 글로벌 모델 수급 (GFS/GEFS 등 GRIB2)
   ├─ (C) WPS (ungrib/metgrid) + (옵션) WRFDA/GSI
   ├─ (D) WRF 실행 (deterministic 또는 소규모 ensemble)
   ├─ (E) Postprocess (station 추출, 일 최고기온 산출, 단위/시간정의 정합)
   ├─ (F) MOS/Calibration (bias-correct + 확률보정)
   ├─ (G) Decision Engine (마켓 bin별 P, 기대값, 리스크 제한)
   ├─ (H) Execution (Kalshi API 주문/취소/로그)
   └─ (I) Verification & Retrain (정산값 수집, 성능/수익률 리포트, 모델 업데이트)
```

- 개인 운영 기준 오케스트레이션은 **Prefect** 우선 권장 (가볍고 빠른 구축).
- 계산/실행 환경은 WRF 컨테이너(Docker/Singularity) + 배치(Slurm/Kubernetes/EC2 Spot) 분리.
- 데이터/로그는 S3(버전 관리) + Postgres(메타데이터)로 분리 운영.

---

## 2) “정산 정의”를 코드로 고정

### 2.1 Kalshi 정산 데이터 정의 (일 최고기온)

- 정산은 "다음날 아침" 확정되는 **NWS Daily Climate Report의 최고기온** 기준.
- DST(일광절약시간) 함정이 핵심:
  - Climate Report는 LST(표준시) 기준.
  - DST 기간에는 **현지 01:00 ~ 익일 00:59**가 기후학적 하루.
- 따라서 모델 산출의 Tmax도 "로컬 자정-자정"이 아니라, **표준시 기준 하루 윈도우**로 계산해야 함.

### 2.2 관측값 소스 자동수집

- 운영/학습 모두 CLI 기반 정합을 맞추는 것이 핵심.
- 직접 전문 파싱 대신 IEM의 구조화 JSON 사용 권장:
  - 예시: `https://mesonet.agron.iastate.edu/json/cli.py?station=KDSM&year=2026`
- CLI 파싱 기반 데이터 소스를 일관되게 쓰면 검증/MOS/정산 정합성이 높아짐.

---

## 3) NWP(WRF) 설계: 일 최고기온 최적화

### 3.1 도메인/해상도 권장

- 단일/소수 도시 타겟:
  - d01: 9 km (배경장/수송)
  - d02: 3 km (도시/해안/지형)
  - 선택 d03: 1 km (열섬/해풍전선이 핵심일 때)
- Tmax는 대류 디테일보다 PBL/복사/구름/토양수분/도시 모수화 영향이 큰 경우가 많음.
- 무조건 1 km 확대보다 **물리옵션 튜닝 + MOS**가 ROI가 좋은 경우가 많음.

### 3.2 물리옵션 출발점

- Radiation: RRTMG (SW/LW)
- LSM: Noah (Noah-MP는 비용 증가)
- PBL: MYNN 또는 YSU (백테스트 성능 비교 후 선택)
- Surface layer: PBL과 일관성 유지
- Microphysics: Thompson 또는 WSM6
- Urban: 도시 영향이 핵심이면 SLUCM/BEP-BEM 검토

운영 원칙:
1. 과거 60~180일 rolling 검증
2. bias 구조 파악
3. MOS가 잘 먹는 physics 조합 선택

### 3.3 경계조건

- deterministic 단일 런만으로는 확률정보가 약함.
- 가능하면 GEFS(또는 multi-IC) 기반 소규모 ensemble(예: 5멤버) 권장.
- ensemble은 uncertainty 보정 및 고신뢰 진입 전략(선별 진입)에 중요.

---

## 4) Postprocess: 관측 정의와 1:1 정합

### 4.1 Station 추출

- WRF 2m T 추출 방식:
  - nearest grid / bilinear / elevation-adjusted
- 권장 시작점: **bilinear + 고도차 보정 피처를 MOS로 학습**
- 고정 lapse-rate(예: 6.5K/km) 하드코딩보다, MOS가 학습하도록 두는 접근 권장.

### 4.2 Climate day 윈도우 Tmax

- 표준시 기준 하루로 window 정의:
  - 비DST: 00:00~23:59 local
  - DST: 01:00~익일 00:59 local
- 시장 단위(대부분 °F)와 최종 산출 단위를 일치.

---

## 0) MOS 우선 원칙

- 이 시스템의 실질적 edge는 WRF raw 값이 아니라 **MOS/Calibration의 확률 품질**에 있음.
- 구현/운영 상세는 `docs/mos.md`를 기준 문서로 참조.

---

## 5) MOS/Calibration: 점예보가 아니라 확률분포

Kalshi 온도 마켓은 bin(구간) 상품이므로 목표는 `P(Tmax ∈ bin_i)`.

### 5.1 EMOS(정규) 권장

- ensemble 기반:
  - `y ~ N(mu, sigma^2)`
  - `mu = a + b * x̄ + seasonal_terms`
  - `sigma^2 = c + d * s^2` (`s^2`: ensemble spread)
- deterministic 기반:
  - `mu = a + b * x`
  - `sigma`는 rolling 오차분산 또는 보조 피처 회귀로 추정

### 5.2 Tmax 특화 피처

- 아침(현지) 기온/이슬점, 초기화 오차
- SWDOWN, cloud fraction, OLR
- 10m wind, PBLH, T2 일변동폭
- 토양수분/토양온도(가능 시)
- 도시/토지피복 클래스, 풍상 수역 여부

### 5.3 추가 Calibration

- EMOS 이후에도 과신/과소신이 남으면
  - isotonic regression 또는 Platt scaling(로지스틱) 후보정
- 목표: reliability diagram의 1:1 근접.

---

## 6) Kalshi 연동: API 기반 운영

### 6.1 공식 API/SDK 사용

- 공식 문서/SDK 사용, Developer Agreement 준수.
- Python SDK는 `kalshi_python_sync`/`kalshi_python_async` 권장.

### 6.2 핵심 엔드포인트

- Orderbook: 현재 호가/유동성
- Candlesticks: 백테스트/리포트용 시계열
- Market: 상태/정산규칙/메타데이터

### 6.3 Decision Engine

1. WRF+MOS로 `P(bin_i)` 계산
2. Kalshi orderbook으로 implied probability/가격 계산
3. 수수료/슬리피지 반영 기대값 산출
4. 진입 조건 예시:
   - `P(bin_i) - implied_prob(bin_i) > edge_threshold`
   - `P(bin_i) >= 0.70`

> 승률 70% 목표는 전체 무차별 진입이 아닌, 고확률 + 양(+)기대값 상황 선별 진입이 전제.

### 6.4 리스크 가드레일

- 일별 최대 노출(USD), 도시별 최대 노출
- 거래 중단 조건: 데이터 수급 실패/WRF 실패/MOS drift
- 저유동성(스프레드 확대) 자동 회피
- 재현성 로깅: 예측분포/가격/주문의사결정 스냅샷 저장

---

## 7) 백테스트 & 지속학습

### 7.1 정산값(관측)

- IEM CLI JSON로 정산 정의 일치 데이터 수집.
- 필요 시 장기 계절성 피처를 NCEI 일자료와 결합.

### 7.2 과거 시장가격

- Candlesticks로 고정 시각(예: 마감 2시간 전) 가격 스냅샷 기반 백테스트.
- Markets/Series 엔드포인트로 자동 탐색.

### 7.3 평가 지표

- 기상: MAE/RMSE, bias, CRPS, reliability
- 트레이딩: hit rate, EV, max drawdown, 거래당 기대값
- 검증 리포트는 station 기준으로 baseline(persistence/climatology/public) vs raw WRF vs MOS를 비교

---

## 8) 구현 체크리스트 (MVP)

- [ ] LST/DST climate-day Tmax 계산 함수 + 테스트
- [ ] IEM CLI 수집 파이프라인 (station/year 배치)
- [ ] WRF postprocess station extractor (bilinear)
- [ ] EMOS 학습/추론 파이프라인
- [ ] bin 확률 계산 + Kalshi 가격 결합 EV 계산기
- [ ] 리스크 가드레일(노출 한도/유동성 필터)
- [ ] 일일 리포트(예측/체결/성과/오류)

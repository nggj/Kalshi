# MOS/Calibration 중심 설계

## 왜 MOS가 핵심인가
Kalshi 온도 마켓은 점예보(`Tmax = 84F`)를 맞추는 문제가 아니라,
구간별 확률 `P(Tmax ∈ bin_i)`의 품질로 수익이 갈립니다.

즉, 원시 WRF 출력만으로는 부족하고 **보정된 확률분포**가 필요합니다.
- 거래 입력은 가격(내재확률) vs 모델 확률의 차이(edge)
- 따라서 핵심은 MAE보다 **calibration + sharpness**

## 파이프라인에서 MOS 입출력
- 입력:
  - station 추출 raw WRF Tmax 특성(단일 또는 ensemble mean/spread)
  - 보조 피처(구름/풍속/PBL/계절 등)
- 처리:
  - EMOS류 모델로 예측분포(`mu`, `sigma`) 생성
  - 분포를 bin에 적분하여 `P(bin)` 계산
  - 필요시 isotonic/Platt로 후보정
- 출력:
  - 각 bin별 calibrated probability vector (합=1)

## 평가 지표: calibration + sharpness
- Reliability: 예측확률과 실제 빈도의 정합
- Brier score: 확률예측의 제곱오차
- Sharpness: 확률질량이 얼마나 날카롭게 집중되는지

운영적으로는 “날카롭고(sharp) 동시에 과신하지 않는(calibrated)” 분포를 목표로 해야
결정 엔진에서 안정적인 양(+)기대값 선별이 가능합니다.

## Station-centric verification과 baseline 비교
MOS의 가치를 보여주려면 같은 station 기준에서 아래를 같이 비교해야 합니다.
- persistence / climatology / public-model CSV baseline
- raw regional model station Tmax
- MOS calibrated probabilities

리포트에서는 MAE/RMSE/bias + Brier score를 함께 보며,
"baseline -> raw model -> MOS" 순으로 개선되는지 확인합니다.

## Quantile MOS (non-normal)
정규분포 하나로는 왜도(skewness)나 두꺼운 꼬리(heavy tail)를 충분히 표현하기 어렵습니다.
이를 보완하기 위해 Quantile Regression MOS를 함께 사용할 수 있습니다.

- 아이디어: 여러 분위수(예: 0.05~0.95)를 직접 예측해 분포를 구성
- 장점: 비대칭 분포를 자연스럽게 표현 가능
- bin 확률 계산: 분위수 기반 CDF 근사 -> bin 경계 CDF 차이

실무적으로는 EMOS(normal)와 Quantile MOS를 나란히 검증하고,
신뢰도/샤프니스/수익성 기준에서 더 안정적인 쪽을 채택합니다.

## WRF + Public stacking
WRF와 Public 모델은 오차 구조가 서로 다를 수 있어, 단일 소스보다 결합했을 때 평균 예측(`mu`)이 안정화될 수 있습니다.

- 아이디어: `mu = f(wrf_tmax, public_tmax, doy_sin, doy_cos)` 형태의 선형 스태킹
- 장점: 특정 날씨 패턴에서 한 모델이 과대/과소편향일 때 다른 모델이 보정
- 실전 포인트: Public 결측은 학습 평균으로 대체해 예측 파이프라인을 끊지 않음
- 출력: Gaussian 분포(`mu`, `sigma`)를 만들고 기존과 동일하게 bin CDF 차이로 `P(bin)` 계산

이 방식은 EMOS 기본선 대비 구조적 보조 신호를 추가하는 경량 확장으로,
검증에서는 MAE/Brier 기준으로 단일 예측기 대비 개선 여부를 확인합니다.

## Hierarchical station/season correction
운영 중에는 관측 체계 변화나 국지 미기상 영향으로 station별 잔여 편향(residual bias)이 천천히 드리프트할 수 있습니다.
이를 완화하기 위해 MOS 분포 예측 뒤에 **계층적(Empirical Bayes) station/season 보정**을 선택적으로 적용할 수 있습니다.

- 입력: 학습 구간의 `residual = y - mu`
- 추정: station bias + month(계절) bias를 shrinkage로 추정
- 핵심: 표본 수가 적은 그룹은 0쪽으로 더 강하게 수축되어 과적합을 줄임
- 적용: 예측 시 분포의 mean 위치를 bias만큼 이동(분산/shape는 유지)

이 방식은 기본 MOS를 바꾸지 않고도 데이터 희소 구간에서 안정적인 편향 보정을 제공해,
out-of-sample drift 구간의 calibration/MAE 악화를 줄이는 데 목적이 있습니다.

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

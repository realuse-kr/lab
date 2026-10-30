# F-4D: 1차 플랜트의 ZOH 이산화와 세 규칙 비교

시간 상수 τ = 0.1초인 1차 플랜트 P(s) = 1 / (τs + 1)을 샘플링 주기 T마다 갱신하는 모델로 바꾼다.
ZOH(zero-order hold, 입력이 한 샘플링 주기 동안 일정하게 유지된다는 가정)와 Forward Euler, Backward Euler, Tustin 규칙을 비교한다.
비교 기준은 입력이 매 샘플 1인 스텝 응답의 샘플 시점 값과 해석해 y(t) = 1 − e^(−t/τ)의 차이(응답 오차 [%])다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `plant_zoh.py` | 본문 코드. 네 규칙의 극점과 최대 응답 오차를 출력한다. |
| `save_results.py` | 검증 계산. 주기별 비교, 구간마다 일정한 임의 입력에 대한 ZOH 검증, 연습문제 답을 `results_f4d.json`에 저장한다. |
| `results_f4d.json` | `save_results.py`의 결과. |

## 실행

```
python plant_zoh.py
python save_results.py
```

필요한 패키지는 NumPy와 SciPy다. 확인한 환경은 Python 3.13, NumPy 2.5.3, SciPy 1.18.1이다.

`plant_zoh.py`의 예상 출력(T = 20 ms)은 다음과 같다.

```
TAU=0.1 s, T=20 ms, exp(-T/TAU)=0.8187
ZOH: 극점 z=0.8187 | 최대 응답 오차 +1.110e-14% (40 ms)
Forward Euler: 극점 z=0.8000 | 최대 응답 오차 +4.020e+00% (100 ms)
Backward Euler: 극점 z=0.8333 | 최대 응답 오차 +1.667e+01% (0 ms)
Tustin: 극점 z=0.8182 | 최대 응답 오차 +9.091e+00% (0 ms)
```

ZOH의 응답 오차 1.110e-14%는 부동소수점 반올림 수준이어서 실행 환경에 따라 자릿수가 조금 달라질 수 있다.

## 값을 바꿔 실습할 때

`plant_zoh.py`의 `TAU` 하나를 바꾸면 이산화와 해석해가 같은 값을 쓴다. 해석해만 따로 바꾸지 않는다. 샘플링 주기는 `main` 안의 `T`와 `step_error`의 인자로 바꾼다.

## 연습문제

1. `TAU`를 0.05로 바꾸고 T = 20 ms에서 ZOH 극점을 구하라.
2. Forward Euler의 극점이 0이 되는 샘플링 주기 T는 얼마인가? `plant_zoh.py`로 확인하라.
3. T = 10 ms에서 Backward Euler의 k = 0 샘플 응답 오차는 몇 %인가?

## 연습문제 답

1. e^(−0.02/0.05) = e^(−0.4) = 0.6703이다.
2. Forward Euler의 극점은 1 − T/τ이므로 T = τ = 0.1초(100 ms)에서 0이다.
3. k = 0 샘플의 응답 오차는 b0 × 100이고 b0 = T/(τ + T) = 0.01/0.11이므로 9.09%다.

## MATLAB

본문의 MATLAB 명령은 MathWorks 문서에서 이름과 인자만 확인했고 실행하지 않았다. 이 폴더의 결과는 모두 Python 실행값이다.

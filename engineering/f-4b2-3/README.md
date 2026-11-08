# F-4B2.3 프리워핑 Tustin으로 옮긴 PI 제어기

플랜트 P(s) = 1/(0.1s + 1)(ZOH 이산화), PI 제어기 C(s) = Kp + Ki/s, Kp = 2, Ki = 40. 이 저장소의 앞선 PI 제어기 계산과 같은 조건이다.

## 파일

- `prewarp_tustin.py`: 프리워핑 없는 Tustin과 프리워핑 Tustin의 계수 (c0, c1), 대입식으로 구한 계수와의 일치 확인, 주파수 응답비, 제어기 출력. 실행: `python prewarp_tustin.py`
- `save_results.py`: 위 모듈로 폐루프 편차와 연습문제 값을 구해 `results_f4b23.json`에 저장한다. 같은 폴더에서 `python save_results.py`로 실행한다. NumPy와 SciPy가 필요하다.
- `results_f4b23.json`: 글에 쓴 숫자의 원본. 기준 각주파수 ω0는 교차 주파수 24.025 rad/s(글에서는 24.0 rad/s로 쓴다)와 60 rad/s다.

## 용어

- 이산 제어기: C_d(z) = (c0·z + c1)/(z − 1), 출력 u[k] = u[k−1] + c0·e[k] + c1·e[k−1]
- 프리워핑 Tustin: s에 a(z − 1)/(z + 1), a = ω0/tan(ω0·T/2)를 대입한다. ω0는 π/T보다 작아야 한다.
- g = tan(ω0·T/2)/(ω0·T/2), c0 = Kp + Ki·T·g/2, c1 = −Kp + Ki·T·g/2
- 주파수 응답비: C_d(exp(jωT))/C(jω)

## 연습문제

1. T = 5 ms, ω0 = 24.025 rad/s일 때 (c0, c1)을 구하라.
2. T = 20 ms, ω0 = 100 rad/s일 때 (c0, c1)을 구하라. ω0가 π/T = 157.08 rad/s보다 작은지도 확인하라.
3. T = 10 ms, ω0 = 24.025 rad/s로 프리워핑한 제어기에 램프 입력 e(t) = t를 넣었을 때 2초에서 제어기 출력 차이(이산 − 연속)를 구하라.

## 풀이

1. g = tan(0.060063)/0.060063 = 1.0012, Ki·T·g/2 = 0.10012이므로 (c0, c1) = (2.1001, −1.8999)
2. g = tan(1.0)/1.0 = 1.5574, Ki·T/2 = 0.4이므로 Ki·T·g/2 = 0.6230이고 (c0, c1) = (2.6230, −1.3770). 100 < 157.08이므로 조건을 만족한다.
3. `save_results.py`가 계산한 값은 0.387이다.

## 확인하지 않은 것

MATLAB은 실행하지 않았다. `c2dOptions('Method','tustin','PrewarpFrequency',w0)`로 만든 c2d 결과가 위 계수와 같은지는 확인하지 않았다. 숫자는 모두 Python(NumPy, SciPy) 실행값이다.

# f-4c-2: PI 제어기를 세 규칙으로 이산화했을 때의 주파수 응답 크기 오차

연속 시간 PI 제어기 C(s) = Kp + Ki/s (Kp = 2, Ki = 40)를 샘플링 주기 T마다 계산하는 C(z) = (b0 z + b1)/(z − 1)로 이산화하는 세 규칙(Forward Euler, Backward Euler, Tustin)을 주파수 응답으로 비교한다. 이산화한 제어기의 주파수 응답은 C(z)에 z = e^(jωT)를 넣어 계산하고, 연속 시간 제어기는 C(jω) = Kp + Ki/(jω)로 계산한다. ω는 각주파수 [rad/s]다.

크기 오차 [%] = (|C(e^(jωT))| / |C(jω)| − 1) × 100이다. 비교 구간은 0.1 rad/s부터 나이퀴스트 주파수 π/T까지이고, 로그 간격 400,001점에서 계산한다. 크기 오차 절댓값 1%를 비교 기준으로 둔다. 이 기준은 비교용이며 일반 설계 기준이 아니다.

플랜트 P(s) = 1/(0.1s + 1)은 이득 교차 주파수(개루프 전달함수 C(s)P(s)의 크기가 1인 주파수)를 구하는 데에만 쓴다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `pi_freq_response.py` | 글 본문의 코드. T = 20 ms에서 세 규칙의 계수, 최대 크기 오차, 크기 오차가 1%를 처음 넘는 주파수를 출력한다. |
| `save_results.py` | `results_f4c2.json`을 만든다. 주기 5·10·20·40 ms 비교, Forward Euler·Backward Euler와 Tustin의 계수 등가 관계, Tustin 적분항만의 크기 오차, 이득 교차 주파수의 크기·위상 차, Ki = 80인 경우, SciPy `freqz`와의 일치, 10배 조밀 격자 결과를 검사한다. |
| `results_f4c2.json` | 본문 표와 문장의 숫자, 확인 항목의 결과. |

## 실행

같은 폴더에서 실행한다.

```
python3 pi_freq_response.py
python3 save_results.py
```

실행 환경: Python 3.13, NumPy 2.5.3, SciPy 1.18.1. `pi_freq_response.py`의 출력은 아래와 같다.

```
T=20ms, 나이퀴스트 주파수 157.08 rad/s
Forward Euler: b0=2.00, b1=-1.20 | 최대 크기 오차 -20.74% (132.0 rad/s) | 1% 처음 넘는 주파수 4.66 rad/s
Backward Euler: b0=2.80, b1=-2.00 | 최대 크기 오차 +19.04% (157.1 rad/s) | 1% 처음 넘는 주파수 4.52 rad/s
Tustin: b0=2.40, b1=-1.60 | 최대 크기 오차 -1.18% (73.6 rad/s) | 1% 처음 넘는 주파수 35.18 rad/s
```

## MATLAB

MATLAB 명령은 제시만 하며 실행하지 않았다. 이 저장소의 숫자는 모두 Python 실행값이다. `c2d`의 방법 이름에는 Forward Euler와 Backward Euler가 없어서 s에 대입하는 식을 z로 직접 적었다.

```
Ts = 0.02; z = tf('z', Ts); Cc = tf([2 40], [1 0]); w = logspace(-1, log10(pi/Ts), 2000);
C = {2 + 40*Ts/(z-1), 2 + 40*Ts*z/(z-1), 2 + 40*(Ts/2)*(z+1)/(z-1)}; % Forward, Backward, Tustin
mc = squeeze(bode(Cc, w)); for k = 1:3, e = (squeeze(bode(C{k}, w))./mc - 1)*100; disp(max(abs(e))), end
```

## 연습문제

1. `pi_freq_response.py`의 `T`를 0.01로 바꾸고 세 규칙의 최대 크기 오차와 1%를 처음 넘는 주파수를 구하라. 나이퀴스트 주파수는 얼마인가.
2. `cont2discrete(([2.4, 40.0], [1, 0]), 0.02, method="euler")`로 Kp 대신 Kp + Ki·T/2 = 2.4를 넣은 Forward Euler 계수를 구하고, Kp = 2인 Tustin 계수와 비교하라.
3. `KI`를 80.0으로 바꾸고 T = 20 ms에서 세 규칙의 최대 크기 오차와 1%를 처음 넘는 주파수를 구하라. 연속 시간 기준 C(jω)도 같은 `KI`를 쓰므로 따로 바꿀 값은 없다.

## 연습문제 정답 (같은 환경에서 실행한 값)

1. 나이퀴스트 주파수는 314.16 rad/s다. 최대 크기 오차는 Forward Euler −10.23%, Backward Euler +9.78%, Tustin −0.31%이고, 1%를 처음 넘는 주파수는 Forward Euler 6.71 rad/s, Backward Euler 6.62 rad/s이며 Tustin은 1%를 넘지 않는다.
2. 두 계수가 모두 b0 = 2.40, b1 = −1.60으로 같다. Forward Euler의 이산 적분기 T/(z − 1)이 Tustin의 이산 적분기 (T/2)(z + 1)/(z − 1)에서 T/2를 뺀 것과 같기 때문이다.
3. 최대 크기 오차는 Forward Euler −41.97%, Backward Euler +35.67%, Tustin −4.14%이고, 1%를 처음 넘는 주파수는 Forward Euler 6.62 rad/s, Backward Euler 6.21 rad/s, Tustin 19.20 rad/s다.

## 이 저장소가 다루지 않는 것

- 위상 차가 폐루프 안정에 주는 영향
- 플랜트 이산화, 계산 지연, 출력 포화, 측정 잡음
- 다른 Kp, Ki 조합 전체(바꿔 볼 수는 있으나 결과를 일반화하지 않는다)
- 실제 기기에서의 측정

라이선스: MIT

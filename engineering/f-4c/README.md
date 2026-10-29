# f-4c: PI 제어기를 세 규칙으로 이산화했을 때의 오버슈트

시상수 0.1초인 1차 플랜트 P(s) = 1/(0.1s + 1)을 PI 제어기 C(s) = Kp + Ki/s (Kp = 2, Ki = 40)로 되먹임해 닫은 폐루프를 대상으로 한다. 제어기를 샘플링 주기 T마다 계산하는 C(z) = (b0 z + b1)/(z − 1)로 이산화하는 세 규칙(Forward Euler, Backward Euler, Tustin)을 비교한다. 플랜트는 영차 유지(zero-order hold)로 정확히 이산화하고 계산 지연은 없다. 스텝 입력의 크기는 1, 계산 구간은 0~1.5초이다.

기준은 연속 시간 폐루프의 스텝 응답 오버슈트 7.91%이다. 오버슈트 오차(%p)는 이산화한 폐루프의 오버슈트에서 이 기준을 뺀 값이다. 안정은 이산화한 폐루프 특성방정식의 근 중 절댓값이 가장 큰 값 max|z|가 1보다 작을 때로 판정한다. 오버슈트 오차는 정확도, max|z|는 안정성을 말하며 서로 다른 기준이다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `pi_discretization.py` | 글 본문의 코드. 주기 5가지에서 세 규칙의 오버슈트, 오차, max\|z\|를 출력한다. |
| `save_results.py` | `results_f4c.json`을 만든다. SciPy `dlsim`과의 일치, 샘플 사이 최댓값, C(z) 계수의 닫힌 식, 작은 주기에서의 수렴, 안정 한계 주기 등 확인 항목을 검사한다. |
| `results_f4c.json` | 본문 표의 숫자와 확인 항목의 결과. |

## 실행

같은 폴더에서 실행한다.

```
python3 pi_discretization.py
python3 save_results.py
```

실행 환경: Python 3.13, NumPy 2.5.3, SciPy 1.18.1. `pi_discretization.py`의 출력은 아래와 같다.

```
연속 시간 오버슈트 7.91%
T=5ms | Forward Euler 9.62% (+1.71%p, 최대|z| 0.929) | Backward Euler 7.95% (+0.05%p, 최대|z| 0.924) | Tustin 8.76% (+0.85%p, 최대|z| 0.927)
T=10ms | Forward Euler 11.70% (+3.79%p, 최대|z| 0.868) | Backward Euler 8.08% (+0.17%p, 최대|z| 0.845) | Tustin 9.77% (+1.86%p, 최대|z| 0.856)
T=20ms | Forward Euler 17.44% (+9.53%p, 최대|z| 0.775) | Backward Euler 8.40% (+0.49%p, 최대|z| 0.675) | Tustin 12.44% (+4.54%p, 최대|z| 0.727)
T=40ms | Forward Euler 37.96% (+30.06%p, 최대|z| 0.734) | Backward Euler 18.68% (+10.78%p, 최대|z| 0.460) | Tustin 21.72% (+13.82%p, 최대|z| 0.524)
T=50ms | Forward Euler 48.32% (+40.41%p, 최대|z| 0.779) | Backward Euler 57.39% (+49.48%p, 최대|z| 0.441) | Tustin 28.99% (+21.09%p, 최대|z| 0.462)
```

## MATLAB

MATLAB 명령은 제시만 하며 실행하지 않았다. 이 저장소의 숫자는 모두 Python 실행값이다. `c2d`의 방법 이름에는 Forward Euler와 Backward Euler가 없어서 s에 대입하는 식을 z로 직접 적었다.

```
Ts = 0.02; z = tf('z', Ts); P = c2d(tf(1, [0.1 1]), Ts, 'zoh');
C = {2 + 40*Ts/(z-1), 2 + 40*Ts*z/(z-1), 2 + 40*(Ts/2)*(z+1)/(z-1)};  % Forward, Backward, Tustin
for k = 1:3, S = stepinfo(feedback(C{k}*P, 1)); disp(S.Overshoot), end
```

## 연습문제

1. `TAU, KP, KI = 0.1, 2.0, 60.0`으로 바꾸고 주기 10 ms와 20 ms에서 세 규칙의 오버슈트와 max|z|를 구하라. 기준 오버슈트는 얼마인가. 계산 구간(`tend`와 기준 응답의 시간축 `t`)이 응답이 수렴할 만큼 긴지도 확인하라.
2. 원래 값(Kp 2, Ki 40)에서 주기를 1 ms 간격으로 키우며 오버슈트 오차가 1%p를 처음 넘기 직전의 주기를 규칙마다 구하라.
3. 같은 값에서 주기를 1 ms 간격으로 키우며 max|z|가 처음 1 이상이 되는 주기를 규칙마다 구하라(`save_results.py`의 `first_unstable`).

## 연습문제 정답 (같은 환경에서 실행한 값)

1. 기준 오버슈트는 14.52%이다. 10 ms에서 Forward Euler 21.97%(max|z| 0.878), Backward Euler 15.27%(0.845), Tustin 18.23%(0.862)이고, 20 ms에서 Forward Euler 33.48%(0.821), Backward Euler 16.83%(0.675), Tustin 23.73%(0.752)이다. 계산 구간 1.5초는 이 값에서도 충분하다.
2. Forward Euler 3 ms, Backward Euler 23 ms, Tustin 5 ms이다.
3. Backward Euler 64 ms, Forward Euler 75 ms, Tustin 110 ms이다.

## 이 저장소가 다루지 않는 것

- 영차 유지 말고 다른 플랜트 이산화, 계산 지연, 출력 포화, 측정 잡음
- 플랜트가 둘 이상이거나 시간상수·이득이 다른 조합 전체(바꿔 볼 수는 있으나 결과를 일반화하지 않는다)
- 규칙마다 오버슈트가 다른 원인 분석
- 실제 기기에서의 측정

라이선스: MIT

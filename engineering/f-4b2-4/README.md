# f-4b2-4: PI 제어기 이산화 여섯 규칙 비교

Kp=2, Ki=40, P(s)=1/(0.1s+1), T=20 ms에서 Forward Euler, Backward Euler, Tustin, 스텝 불변, 극·영점 매칭, 프리워핑 Tustin(ω0=24 rad/s)의 계수·크기비·출력 차이·폐루프 편차를 계산한다.

- `python3 six_rules.py`: 계수, ωc 크기비·위상차, 스텝·램프 차이, 비율별 편차 출력
- `python3 save_results.py`: results_f4b24.json 저장, 네 규칙의 계수를 SciPy cont2discrete와 대조

## 연습문제
1. KP를 3으로 바꾸면 Forward Euler와 스텝 불변의 계수는 계속 같은가?
2. W0_PREWARP를 10 rad/s로 바꾸면 낮은 주파수 적분 이득은 얼마인가?
3. 비율 6에서 편차가 가장 작은 규칙은 어느 것인가?

## 풀이
1. 같다. 두 규칙 모두 (KP, -KP + KI·T)다.
2. g = tan(0.1)/0.1 = 1.0033이므로 40 × 1.0033 ≈ 40.13이다.
3. Tustin(24.3%)이다.

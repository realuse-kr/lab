# f-4b2-2: 극·영점 매칭으로 옮긴 PI 제어기

조건: 플랜트 P(s)=1/(0.1s+1), PI 제어기 C(s)=Kp+Ki/s, Kp=2, Ki=40.

- matched_pz.py: 극·영점 매칭 계수와 네 규칙(극·영점 매칭, 스텝 불변, Backward Euler, Tustin)의 제어기 출력 차이
- save_results.py: 폐루프 편차, 교차 주파수 비교, 연습문제 값을 계산해 results_f4b22.json에 저장
- 실행: `python matched_pz.py`, `python save_results.py` (Python 3, NumPy, SciPy)

극·영점 매칭의 이득은 낮은 주파수 적분 이득을 Ki로 맞추는 규칙(K = Ki·T/(1−exp(−Ki·T/Kp)))이며 MATLAB c2d의 matched 방식과 같은 규칙인지는 확인하지 않았다. MATLAB은 실행하지 않았다.

## 연습문제
1. T = 5 ms에서 극·영점 매칭 계수 (c0, c1)을 구하라.
2. Ki = 20, T = 20 ms에서 계수 (c0, c1)을 구하라.
3. T = 10 ms에서 램프 입력 e=t의 2초 시점 제어기 출력 차이를 구하라.

## 해답
1. (2.1017, −1.9017)
2. (2.2067, −1.8067)
3. +0.0133

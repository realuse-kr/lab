# F-5C 본문 코드: 세 방식(한도 없음, 한도와 적분 그대로, 한도와 적분 멈춤)을 돌려 지표를 출력한다.
import numpy as np
kp, ki, ts, tau, dr, umax = 2.0, 6.0, 1e-3, 0.5, 8.0, 10.0
a = np.exp(-ts / tau)  # 한 샘플 동안 이전 측정값이 남는 비율(따라가는 비율은 1 - a)
k0, n = 100, 4000      # 설정값이 바뀌는 샘플 번호, 샘플 수
for mode in ("한도 없음", "한도, 적분 그대로", "한도, 적분 멈춤"):
    y, u, integ, i_max = np.zeros(n), np.zeros(n), 0.0, 0.0
    for k in range(n):
        e = (dr if k >= k0 else 0.0) - y[k]
        u_try = kp * e + integ + ki * ts * e
        if not (mode == "한도, 적분 멈춤" and abs(u_try) > umax and u_try * e > 0):
            integ += ki * ts * e
        u[k], i_max = kp * e + integ, max(i_max, integ)
        us = u[k] if mode == "한도 없음" else min(max(u[k], -umax), umax)
        if k + 1 < n:
            y[k + 1] = a * y[k] + (1 - a) * us
    print(f"{mode}: 오버슈트 {y.max() - dr:.3f}, 한도를 넘은 시간 {np.sum(u > umax)} ms, 적분 항 최댓값 {i_max:.2f}")
tr = tau * np.log(umax / (umax - dr))
print(f"식 Ki×A = {ki * (tau * dr - (umax - dr) * tr):.2f}")

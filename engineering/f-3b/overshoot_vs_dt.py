"""적분 간격(dt)에 따라 스텝 응답 오버슈트가 얼마나 달라지는지 본다.

가상 2차 시스템: 고유각주파수 WN(rad/s), 감쇠비 ZETA. 계단 입력 u = 1.
TOL은 예시 허용 오차(%p)이며 독자의 기준은 다를 수 있다.
EXACT는 2차 시스템 계단 응답의 해석 오버슈트(%)다.
"""
import numpy as np

WN, ZETA = 10.0, 0.3
T_END = 2.0
TOL = 1.0
EXACT = 100 * np.exp(-np.pi * ZETA / np.sqrt(1 - ZETA**2))

print(f"해석 오버슈트 = {EXACT:.3f}%")
for dt in [0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.02, 0.04]:
    y, v, peak = 0.0, 0.0, 0.0
    for _ in range(int(round(T_END / dt))):
        a = WN**2 * (1.0 - y) - 2 * ZETA * WN * v
        y, v = y + dt * v, v + dt * a
        peak = max(peak, y)
    err = 100 * (peak - 1.0) - EXACT
    verdict = "허용 이내" if abs(err) <= TOL else "허용 초과"
    print(f"dt={dt * 1000:5.1f}ms  오버슈트={EXACT + err:6.2f}%  "
          f"오차={err:+6.2f}%p  {verdict}")

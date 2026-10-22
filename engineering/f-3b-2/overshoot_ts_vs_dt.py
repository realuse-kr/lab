"""제어기 주기(TS)와 시뮬레이션 간격(dt)을 따로 정하면 오버슈트가 어떻게 읽히는지 본다.

가상 2차 시스템에 비례 제어기(KP)를 붙인다. 제어기는 TS마다 출력을 한 번 정하고
다음 주기까지 그대로 유지한다. dt = TS이면 두 값이 같고, dt = TS/100이면 간격을 따로 잘게 잡은 것이다.
EXACT는 주기 없이 연속으로 동작하는 제어기의 오버슈트를 수식으로 구한 값(%)이다.
"""
import numpy as np

WN, ZETA, KP, T_END = 10.0, 0.3, 3.0, 2.0
ZETA_CL = ZETA / np.sqrt(1 + KP)
EXACT = 100 * np.exp(-np.pi * ZETA_CL / np.sqrt(1 - ZETA_CL**2))
Y_END = KP / (1 + KP)

print(f"연속 제어기 오버슈트(수식) = {EXACT:.2f}%")
for ts in [0.001, 0.002, 0.005, 0.01]:
    out = []
    for dt in [ts, ts / 100]:
        y, v, peak = 0.0, 0.0, 0.0
        for _ in range(int(round(T_END / ts))):
            u = KP * (1.0 - y)
            for _ in range(int(round(ts / dt))):
                a = WN**2 * (u - y) - 2 * ZETA * WN * v
                y, v = y + dt * v, v + dt * a
                peak = max(peak, y)
        out.append(100 * (peak / Y_END - 1.0))
    print(f"TS={ts * 1000:4.1f}ms dt=TS:{out[0]:7.2f}% dt=TS/100:{out[1]:7.2f}%")

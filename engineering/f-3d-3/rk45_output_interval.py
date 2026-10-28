import numpy as np
from scipy.integrate import solve_ivp

TAU, T = 0.010, 0.060            # 시간상수와 계산 구간(초)
EXACT = np.exp(-T / TAU)         # 해석해 x(T) = e^(-6)
f = lambda t, x: -x / TAU        # dx/dt = -x/τ
KW = dict(method="RK45", rtol=1e-6, atol=1e-12)

def rk45(dt_out, how):
    if how == "t_eval":          # 출력 시각만 지정. 시간 스텝 크기는 RK45가 정한다
        t_out = np.linspace(0, T, round(T / dt_out) + 1)
        s = solve_ivp(f, (0, T), [1.0], t_eval=t_out, **KW)
    else:                        # 출력 간격을 시간 스텝 크기의 상한으로 지정
        s = solve_ivp(f, (0, T), [1.0], max_step=dt_out, **KW)
    return s.nfev, abs(s.y[0, -1] - EXACT) / EXACT

def rk4(h):                      # 고정 시간 스텝: 출력 간격 = 시간 스텝 크기 h
    x = 1.0
    for _ in range(round(T / h)):
        k1 = f(0, x); k2 = f(0, x + h / 2 * k1)
        k3 = f(0, x + h / 2 * k2); k4 = f(0, x + h * k3)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return 4 * round(T / h), abs(x - EXACT) / EXACT

if __name__ == "__main__":
    for dt in (0.010, 0.005, 0.002, 0.001, 0.0005, 0.0001):
        a, b, c = rk45(dt, "t_eval"), rk45(dt, "max_step"), rk4(dt)
        print(f"출력 간격 {dt*1e3:g}ms | t_eval {a[0]}회 {a[1]:.1e}"
              f" | max_step {b[0]}회 {b[1]:.1e} | RK4 {c[0]}회 {c[1]:.1e}")

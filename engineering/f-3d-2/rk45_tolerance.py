import numpy as np
from scipy.integrate import solve_ivp

TAU, T = 0.010, 0.060            # 시간상수와 계산 구간(초)
EXACT = np.exp(-T / TAU)         # 해석해 x(T) = e^(-6)

def run(rtol, atol):
    s = solve_ivp(lambda t, x: -x / TAU, (0, T), [1.0],
                  method="RK45", rtol=rtol, atol=atol)
    h = np.diff(s.t)             # 수용된 시간 스텝 크기
    err = abs(s.y[0, -1] - EXACT) / EXACT
    return s.nfev, h, err

if __name__ == "__main__":
    for atol in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2):
        nfev, h, err = run(1e-6, atol)
        print(f"atol={atol:g} 함수 평가 {nfev}회 시간 스텝 {len(h)}개 오차 {err:.1e}")

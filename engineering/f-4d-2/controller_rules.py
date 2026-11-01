import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
TAU, KP, KI = 0.1, 2.0, 40.0  # 플랜트 P(s) = 1/(TAU*s + 1), PI 제어기 C(s) = KP + KI/s
RULES = {"ZOH": "zoh", "Forward Euler": "euler", "Backward Euler": "backward_diff", "Tustin": "bilinear"}

def controller(rule, T):  # C(z) = (c0 z + c1) / (z + d1) -> (c0, c1, d1)
    num, den, _ = cont2discrete(([KP, KI], [1.0, 0.0]), T, method=rule)
    return np.squeeze(num)[0], np.squeeze(num)[1], np.squeeze(den)[1]

def deviation_pct(rule, T, t_end=1.0):  # 연속 폐루프 대비 이산 폐루프의 샘플 시점 편차 [% of 최종값 1]
    t = np.arange(int(round(t_end / T)) + 1) * T
    a, (c0, c1, d1) = np.exp(-T / TAU), controller(rule, T)  # 플랜트 ZOH 모델: y[k+1] = a*y[k] + (1-a)*u[k]
    y, e_old, u_old = [0.0], 0.0, 0.0
    for _ in t[1:]:
        e = 1.0 - y[-1]  # 오차 = 목표값 1 - 출력
        u = c0 * e + c1 * e_old - d1 * u_old  # 제어기 차분방정식
        y.append(a * y[-1] + (1 - a) * u)
        e_old, u_old = e, u
    f = lambda _, s: [(-s[0] + KP * (1 - s[0]) + KI * s[1]) / TAU, 1 - s[0]]  # 연속 폐루프, s = [출력, 오차의 적분]
    y_c = solve_ivp(f, (0, t[-1]), [0, 0], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14).y[0]
    return (np.array(y) - y_c) * 100

if __name__ == "__main__":
    for name, r in RULES.items():
        e, (c0, c1, d1) = deviation_pct(r, 0.02), controller(r, 0.02)
        print(f"{name}: c0={c0:.2f} c1={c1:.2f} d1={d1:+.2f} | 최대 편차 {e[np.argmax(abs(e))]:+.2f}%")

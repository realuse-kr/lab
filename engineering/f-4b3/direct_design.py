"""F-4B3: PI 제어기를 이산 시간 플랜트 모델 위에서 직접 설계하고, 연속 시간에서 설계해 이산화한 여섯 규칙(F-4B2.4)과 폐루프 극점·폐루프 편차를 같은 조건에서 비교한다.

플랜트 P(s) = 1/(TAU s + 1), 연속 시간 PI 제어기 C(s) = KP + KI/s(이산화 규칙의 출발점), 샘플링 주기 T = 20 ms.
이산 제어기는 C_d(z) = (c0 z + c1)/(z - 1)이고, 출력은 u[k] = u[k-1] + c0 e[k] + c1 e[k-1]이다.
"""
import numpy as np
from scipy.signal import TransferFunction, cont2discrete, step as cstep, zpk2tf

KP, KI, TAU = 2.0, 40.0, 0.1
A = KI / KP  # C(s)의 영점 s = -A
W0_PREWARP = 24.0  # 프리워핑 각주파수 [rad/s]
T_DEFAULT = 0.02  # 샘플링 주기 [s]


def target_poles(T):
    """연속 시간 폐루프의 극점 두 개를 z = exp(sT)로 옮긴 목표 극점. 연속 폐루프 특성방정식은 TAU s^2 + (1 + KP) s + KI = 0."""
    s = np.roots([TAU, 1 + KP, KI])
    return np.exp(s * T)


def direct_design(T):
    """이산 폐루프의 특성방정식 (z - 1)(z - alpha) + (1 - alpha)(c0 z + c1)을 목표 다항식 z^2 + a1 z + a0과 같게 놓고 (c0, c1)을 푼다."""
    alpha = np.exp(-T / TAU)  # 플랜트를 ZOH로 이산화한 P(z) = (1 - alpha)/(z - alpha)
    zp = target_poles(T)
    a1 = -(zp[0] + zp[1]).real
    a0 = (zp[0] * zp[1]).real
    c0 = (a1 + 1 + alpha) / (1 - alpha)
    c1 = (a0 - alpha) / (1 - alpha)
    return float(c0), float(c1)


def emulation_rules(T, w0=W0_PREWARP):
    """연속 시간 C(s)를 이산화하는 여섯 규칙의 (c0, c1). F-4B2.4의 six_rules.py와 같은 식이다."""
    p = np.exp(-A * T)
    k = KI * T / (1 - p)
    num, den = zpk2tf([p], [1.0], k)
    assert np.allclose(den, [1.0, -1.0])
    x = w0 * T / 2
    g = np.tan(x) / x
    return {
        "Forward Euler": (KP, -KP + KI * T),
        "Backward Euler": (KP + KI * T, -KP),
        "Tustin": (KP + KI * T / 2, -KP + KI * T / 2),
        "스텝 불변": (KP, -KP + KI * T),
        "극·영점 매칭": (float(num[0]), float(num[1])),
        "프리워핑 Tustin": (KP + KI * T * g / 2, -KP + KI * T * g / 2),
    }


def closed_loop_poles(c, T):
    """이산 폐루프 특성다항식의 근. 플랜트는 scipy의 cont2discrete(ZOH)로 이산화해 쓴다."""
    pn, pd, _ = cont2discrete(([1.0], [TAU, 1.0]), T, "zoh")[:3]
    pn = np.trim_zeros(np.ravel(pn), "f")
    open_num = np.polymul([c[0], c[1]], pn)
    open_den = np.polymul([1.0, -1.0], np.ravel(pd))
    n = max(len(open_num), len(open_den))
    poly = np.pad(open_den, (n - len(open_den), 0)) + np.pad(open_num, (n - len(open_num), 0))
    return np.roots(poly)


def pole_distance(c, T):
    """폐루프 극점 두 개와 목표 극점 두 개를 거리가 가장 짧게 짝지었을 때 거리의 최댓값."""
    r = closed_loop_poles(c, T)
    zt = target_poles(T)
    pair_a = max(abs(r[0] - zt[0]), abs(r[1] - zt[1]))
    pair_b = max(abs(r[0] - zt[1]), abs(r[1] - zt[0]))
    return float(min(pair_a, pair_b))


def closed_loop_deviation(c, T, t_end=2.0):
    """이산 폐루프 스텝 응답과 연속 시간 폐루프 스텝 응답의 샘플 시점 차이의 최댓값(2초에 가장 가까운 샘플 시점까지)."""
    n = int(round(t_end / T))
    alpha = np.exp(-T / TAU)
    y, u, e_old = 0.0, 0.0, 0.0
    ys = []
    for _ in range(n + 1):
        ys.append(y)
        e = 1.0 - y
        u = u + c[0] * e + c[1] * e_old
        e_old = e
        y = alpha * y + (1 - alpha) * u
    sys_cl = TransferFunction([KP, KI], [TAU, 1 + KP, KI])
    t = np.arange(n + 1) * T
    _, yc = cstep(sys_cl, T=t)
    return float(np.max(np.abs(np.array(ys) - yc)))


if __name__ == "__main__":
    T = T_DEFAULT
    s = np.roots([TAU, 1 + KP, KI])
    zt = target_poles(T)
    print(f"연속 폐루프 극점 {s[0]:.2f} rad/s, 목표 극점 z = {zt[0]:.4f}, |z| = {abs(zt[0]):.4f}")
    c_direct = direct_design(T)
    print(f"직접 이산 설계: c0={c_direct[0]:.4f} c1={c_direct[1]:+.4f} 영점={-c_direct[1] / c_direct[0]:.4f} 적분 이득={(c_direct[0] + c_direct[1]) / T:.2f}")
    rows = dict(emulation_rules(T))
    rows["직접 이산 설계"] = c_direct
    for name, c in rows.items():
        r = sorted(closed_loop_poles(c, T), key=lambda x: x.imag)[1]
        print(f"{name}: 폐루프 극점 {r.real:.4f}{r.imag:+.4f}j 목표까지 거리 {pole_distance(c, T):.4f} 폐루프 편차 {closed_loop_deviation(c, T) * 100:.1f}%")

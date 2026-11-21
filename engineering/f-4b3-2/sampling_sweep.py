"""F-4B3.2: 샘플링 주기 T를 바꿔 가며 에뮬레이션 여섯 규칙과 직접 이산 설계의 폐루프 편차를 비교한다.

플랜트 P(s) = 1/(TAU s + 1), 연속 시간 PI 제어기 C(s) = KP + KI/s(에뮬레이션 규칙의 출발점).
이산 제어기는 C_d(z) = (c0 z + c1)/(z - 1)이고, 출력은 u[k] = u[k-1] + c0 e[k] + c1 e[k-1]이다.
목표 극점은 연속 시간 폐루프 극점을 z = exp(sT)로 옮긴 값이며 T마다 새로 계산한다.
"""
import numpy as np
from scipy.signal import TransferFunction, cont2discrete, step as cstep, zpk2tf

KP, KI, TAU = 2.0, 40.0, 0.1
A = KI / KP  # C(s)의 영점 s = -A
W0_PREWARP = 24.0  # 프리워핑 각주파수 [rad/s]. KP, KI, TAU를 바꾸면 함께 바꾼다.
DIRECT = "직접 이산 설계"
EMULATION = ["Forward Euler", "Backward Euler", "Tustin", "스텝 불변", "극·영점 매칭", "프리워핑 Tustin"]
T_MS_MAX = 120  # 샘플링 주기를 1 ms부터 이 값까지 1 ms 단위로 계산한다
TOLS = (0.05, 0.10, 0.20)  # 폐루프 편차 한도(목표값 1에 대한 비율)


def target_poles(T):
    """연속 시간 폐루프 극점 두 개를 z = exp(sT)로 옮긴 목표 극점. 연속 폐루프 특성방정식은 TAU s^2 + (1 + KP) s + KI = 0."""
    return np.exp(np.roots([TAU, 1 + KP, KI]) * T)


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
    """연속 시간 C(s)를 이산화하는 여섯 규칙의 (c0, c1). F-4B3의 direct_design.py와 같은 식이다."""
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


def coefficients(rule, T):
    return direct_design(T) if rule == DIRECT else emulation_rules(T)[rule]


def closed_loop_poles(c, T):
    """이산 폐루프 특성다항식의 근. 플랜트는 scipy의 cont2discrete(ZOH)로 이산화해 쓴다."""
    pn, pd, _ = cont2discrete(([1.0], [TAU, 1.0]), T, "zoh")[:3]
    pn = np.trim_zeros(np.ravel(pn), "f")
    open_num = np.polymul([c[0], c[1]], pn)
    open_den = np.polymul([1.0, -1.0], np.ravel(pd))
    n = max(len(open_num), len(open_den))
    poly = np.pad(open_den, (n - len(open_den), 0)) + np.pad(open_num, (n - len(open_num), 0))
    return np.roots(poly)


def pole_abs_max(c, T):
    """이산 폐루프 극점 중 크기 |z|가 가장 큰 값. 1 이상이면 불안정이다."""
    return float(max(abs(closed_loop_poles(c, T))))


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


def longest_period(rule, tol, t_ms_max=T_MS_MAX):
    """샘플링 주기를 1 ms부터 늘려 가며 폐루프 편차가 처음 tol을 넘기 직전의 주기[ms]를 구한다."""
    for t_ms in range(1, t_ms_max + 1):
        T = t_ms / 1000
        if closed_loop_deviation(coefficients(rule, T), T) > tol:
            return t_ms - 1
    return t_ms_max


def first_unstable(rule, t_ms_max=T_MS_MAX):
    """폐루프 극점 크기가 처음 1 이상이 되는 샘플링 주기[ms]. 없으면 None."""
    for t_ms in range(1, t_ms_max + 1):
        T = t_ms / 1000
        if pole_abs_max(coefficients(rule, T), T) >= 1.0:
            return t_ms
    return None


if __name__ == "__main__":
    rules = EMULATION + [DIRECT]
    print("연속 폐루프 극점 s =", np.round(np.roots([TAU, 1 + KP, KI]), 2)[0], "rad/s, 목표 극점 크기 |z| = exp(-15 T)")
    print("샘플링 주기별 폐루프 편차 (불안정은 폐루프 극점 크기가 1 이상)")
    for t_ms in (5, 10, 20, 40, 80):
        T = t_ms / 1000
        cells = []
        for r in rules:
            c = coefficients(r, T)
            cells.append("불안정" if pole_abs_max(c, T) >= 1.0 else f"{closed_loop_deviation(c, T) * 100:.1f}%")
        print(f"T={t_ms} ms: " + " | ".join(f"{r} {v}" for r, v in zip(rules, cells)))
    print("폐루프 편차가 처음 한도를 넘기 직전의 샘플링 주기 [ms]")
    for tol in TOLS:
        print(f"한도 {tol:.0%}: " + " | ".join(f"{r} {longest_period(r, tol)}" for r in rules))
    print("폐루프 극점 크기가 처음 1 이상이 되는 샘플링 주기 [ms]")
    print(" | ".join(f"{r} {first_unstable(r) or '없음'}" for r in rules))

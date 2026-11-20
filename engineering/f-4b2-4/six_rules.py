"""F-4B2.4: PI 제어기를 이산화하는 여섯 규칙의 계수·주파수 응답비·출력 차이·폐루프 편차를 같은 조건에서 계산한다."""
import numpy as np
from scipy.signal import zpk2tf, TransferFunction, step as cstep

KP, KI, TAU = 2.0, 40.0, 0.1  # C(s) = KP + KI/s, 플랜트 P(s) = 1/(TAU s + 1)
A = KI / KP  # 영점 s = -A
W0_PREWARP = 24.0  # 프리워핑 각주파수 [rad/s], 교차 주파수 부근


def crossover_and_bandwidth():
    """연속 시간 개루프 L = C P의 교차 주파수 [rad/s]와 폐루프 대역폭 [Hz]를 이분법으로 구한다."""
    L = lambda w: (KP + KI / (1j * w)) / (1j * w * TAU + 1)
    Tcl = lambda w: L(w) / (1 + L(w))
    def bisect(f, lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)
    wc = bisect(lambda w: abs(L(w)) - 1.0, 1.0, 200.0)
    wb = bisect(lambda w: abs(Tcl(w)) - 1 / np.sqrt(2), 1.0, 200.0)
    return wc, wb / (2 * np.pi)


def rules(T, w0=W0_PREWARP):
    """C(z) = (c0 z + c1)/(z - 1)의 (c0, c1)을 규칙마다 돌려준다."""
    p = np.exp(-A * T)  # 영점 s=-A -> z=exp(-A T), 극점 s=0 -> z=1
    k = KI * T / (1 - p)  # 낮은 주파수 적분 이득을 KI로 맞춘다
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


def ratio(c, T, w):
    z = np.exp(1j * w * T)
    return (c[0] * z + c[1]) / (z - 1) / (KP + KI / (1j * w))


def controller_out(c, e):
    u, e_old, u_old = [], 0.0, 0.0
    for ek in e:
        u_old = c[0] * ek + c[1] * e_old + u_old
        e_old = ek
        u.append(u_old)
    return np.array(u)


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
    wc, fb = crossover_and_bandwidth()
    T = 0.02
    t = np.arange(101) * T  # 처음 2초의 샘플 시점
    print(f"교차 주파수 {wc:.3f} rad/s, 폐루프 대역폭 {fb:.4f} Hz, T = 20 ms의 샘플링 비율 {1 / (T * fb):.2f}")
    for name, c in rules(T).items():
        r = ratio(c, T, wc)
        step = controller_out(c, np.ones(t.size)) - (KP + KI * t)  # 스텝 입력 e = 1
        ramp = controller_out(c, t) - (KP * t + KI * t ** 2 / 2)  # 램프 입력 e = t
        print(f"{name}: c0={c[0]:.4f} c1={c[1]:+.4f} 영점={-c[1] / c[0]:.4f} 적분 이득={(c[0] + c[1]) / T:.2f}"
              f" | 크기비 {abs(r):.3f} 위상차 {np.degrees(np.angle(r)):+.2f}도"
              f" | 스텝 입력 차이 {step[-1]:+.3f} 램프 입력 차이 {ramp[-1]:+.3f} | 폐루프 편차 {closed_loop_deviation(c, T) * 100:.1f}%")
    for m in (6, 8, 10, 20, 25, 30, 40, 100):
        Tm = 1 / (m * fb)
        row = {n: closed_loop_deviation(c, Tm) * 100 for n, c in rules(Tm).items()}
        print(f"비율 {m} ({Tm * 1e3:.1f} ms): " + " ".join(f"{n} {v:.1f}%" for n, v in row.items()))

"""F-4B4: 한 샘플 지연이 위상 여유를 얼마나 줄이는지 계산한다.

플랜트 P(s) = 1/(TAU s + 1), PI 제어기 C(s) = KP + KI/s, 개루프 L(s) = C(s) P(s).
연속 시간 루프의 지연은 e^(-sT)로, 이산 루프의 한 샘플 지연은 z^-1로 개루프에 곱한다.
이산 루프는 플랜트를 ZOH로 이산화한 P(z) = (1 - alpha)/(z - alpha), alpha = exp(-T/TAU)와
Tustin으로 이산화한 C_d(z) = (c0 z + c1)/(z - 1), c0 = KP + KI T/2, c1 = -KP + KI T/2를 쓴다.
위상 여유는 개루프 크기가 1이 되는 각주파수에서 개루프 위상에 180도를 더한 값이다.
"""
import numpy as np
from scipy.optimize import brentq

KP, KI, TAU = 2.0, 40.0, 0.1
W_LO, W_HI = 1.0, 200.0  # 개루프 크기가 1이 되는 각주파수를 찾는 구간 [rad/s]. KP, KI, TAU를 바꾸면 이 구간 안에 교차점이 있어야 한다.


def loop_cont(w, T=0.0):
    """연속 시간 개루프 L(jw)에 지연 T[s]를 넣은 값."""
    s = 1j * w
    return (KP + KI / s) / (TAU * s + 1) * np.exp(-s * T)


def loop_disc(w, T, delays=1):
    """이산 개루프 L_d(e^(jwT)) z^-delays. delays는 지연 샘플 수(0이면 지연 없음)."""
    z = np.exp(1j * w * T)
    al = np.exp(-T / TAU)
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    return (c0 * z + c1) / (z - 1) * (1 - al) / (z - al) * z ** (-delays)


def crossover_and_margin(loop, w_hi):
    """개루프 크기가 1이 되는 각주파수 wc[rad/s]와 위상 여유[도].

    위상은 저주파에서 고주파로 촘촘한 격자를 따라 이어 붙인(unwrap) 값을 쓰고 wc에서 보간한다.
    그래서 지연이 커서 위상이 -180도를 지나도 위상 여유가 음수로 나온다.
    """
    assert abs(loop(W_LO)) > 1.0 > abs(loop(w_hi)), "교차점이 구간 밖이다"
    wc = brentq(lambda w: abs(loop(w)) - 1.0, W_LO, w_hi, xtol=1e-12)
    grid = np.linspace(0.01, w_hi, 400001)
    phase = np.unwrap(np.angle(loop(grid)))
    return float(wc), float(180.0 + np.degrees(np.interp(wc, grid, phase)))


def margin_cont(T):
    """지연 T[s]를 넣은 연속 시간 루프의 (wc, 위상 여유[도])."""
    return crossover_and_margin(lambda w: loop_cont(w, T), W_HI)


def margin_disc(T, delays):
    """샘플링 주기 T[s], 지연 샘플 수 delays인 이산 루프의 (wc, 위상 여유[도])."""
    return crossover_and_margin(lambda w: loop_disc(w, T, delays), 0.999 * np.pi / T)


def disc_pole_abs_max(T, delays):
    """이산 폐루프 극점 중 크기가 가장 큰 값. 1 이상이면 불안정이다.
    특성다항식은 (z - 1)(z - alpha) z^delays + (1 - alpha)(c0 z + c1)이다."""
    al = np.exp(-T / TAU)
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    den = np.polymul(np.polymul([1.0, -1.0], [1.0, -al]), [1.0] + [0.0] * delays)
    num = (1 - al) * np.array([c0, c1])
    n = max(len(den), len(num))
    poly = np.pad(den, (n - len(den), 0)) + np.pad(num, (n - len(num), 0))
    return float(max(abs(np.roots(poly))))


if __name__ == "__main__":
    wc, pm0 = margin_cont(0.0)
    print(f"연속 시간 개루프: 교차 각주파수 wc = {wc:.2f} rad/s, 위상 여유 {pm0:.1f}도, "
          f"지연 마진 {np.radians(pm0) / wc * 1000:.1f} ms")
    print("연속 시간 루프에 지연 T를 넣었을 때 (줄어든 각도는 위상 여유 72.8도에서 뺀 값, wc*T는 식의 값)")
    for t_ms in (5, 10, 20, 40, 60):
        pm = margin_cont(t_ms / 1000)[1]
        print(f"T={t_ms} ms: 위상 여유 {pm:.1f}도 | 줄어든 각도 {pm0 - pm:.1f}도 | wc*T {np.degrees(wc * t_ms / 1000):.1f}도")
    print("이산 루프(Tustin + ZOH 플랜트)에 한 샘플 지연을 넣었을 때 (안정은 폐루프 극점 크기가 모두 1 미만)")
    for t_ms in (5, 10, 20, 40):
        T = t_ms / 1000
        w0, pm_no = margin_disc(T, 0)
        w1, pm_d = margin_disc(T, 1)
        stable = "안정" if disc_pole_abs_max(T, 1) < 1.0 else "불안정"
        print(f"T={t_ms} ms: 지연 없음 {pm_no:.1f}도 | 한 샘플 지연 {pm_d:.1f}도 | 줄어든 각도 {pm_no - pm_d:.1f}도 | "
              f"지연 없는 루프의 wc*T {np.degrees(w0 * T):.1f}도 | 한 샘플 지연 폐루프 {stable}")
    first = next(t for t in range(1, 41) if disc_pole_abs_max(t / 1000, 1) >= 1.0)
    print(f"한 샘플 지연 폐루프의 극점 크기가 처음 1 이상이 되는 샘플링 주기 {first} ms")

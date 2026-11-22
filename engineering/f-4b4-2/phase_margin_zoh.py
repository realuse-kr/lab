"""F-4B4.2: 영차 유지(ZOH)가 개루프 위상을 얼마나 낮추는지, 위상 여유로 계산한다.

플랜트 P(s) = 1/(TAU s + 1), PI 제어기 C(s) = KP + KI/s, 개루프 L(s) = C(s) P(s).
ZOH의 주파수 응답은 (1 - e^(-jwT))/(jw)이고, 아래에서는 크기를 T로 나눠 (1 - e^(-jwT))/(jwT)를 쓴다.
이산 루프는 플랜트를 ZOH로 이산화한 P(z) = (1 - alpha)/(z - alpha), alpha = exp(-T/TAU)와
Tustin으로 이산화한 C_d(z) = (c0 z + c1)/(z - 1), c0 = KP + KI T/2, c1 = -KP + KI T/2를 쓴다(F-4B4와 같다).
위상 여유는 개루프 크기가 1이 되는 각주파수에서 개루프 위상에 180도를 더한 값이다.
"""
import numpy as np
from scipy.optimize import brentq

KP, KI, TAU = 2.0, 40.0, 0.1
W_LO, W_HI = 1.0, 200.0  # 개루프 크기가 1이 되는 각주파수를 찾는 구간 [rad/s]. KP, KI, TAU를 바꾸면 이 구간 안에 교차점이 있어야 한다.


def zoh(w, T):
    """ZOH의 주파수 응답을 T로 나눈 값 (1 - e^(-jwT))/(jwT)."""
    x = 1j * w * T
    return (1 - np.exp(-x)) / x


def loop_base(w):
    """연속 시간 개루프 L(jw)."""
    s = 1j * w
    return (KP + KI / s) / (TAU * s + 1)


def loop_zoh(w, T):
    """연속 시간 개루프에 ZOH를 곱한 값."""
    return loop_base(w) * zoh(w, T)


def loop_delay(w, T):
    """연속 시간 개루프에 길이 T[s]인 지연 e^(-jwT)를 곱한 값."""
    return loop_base(w) * np.exp(-1j * w * T)


def loop_disc(w, T):
    """지연 없는 이산 개루프 L_d(e^(jwT)). 플랜트는 ZOH로, 제어기는 Tustin으로 이산화했다."""
    z = np.exp(1j * w * T)
    al = np.exp(-T / TAU)
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    return (c0 * z + c1) / (z - 1) * (1 - al) / (z - al)


def crossover_and_margin(loop, w_hi):
    """개루프 크기가 1이 되는 각주파수 wc[rad/s]와 위상 여유[도].

    위상은 저주파에서 고주파로 촘촘한 격자를 따라 이어 붙인(unwrap) 값을 쓰고 wc에서 보간한다.
    """
    assert abs(loop(W_LO)) > 1.0 > abs(loop(w_hi)), "교차점이 구간 밖이다"
    wc = brentq(lambda w: abs(loop(w)) - 1.0, W_LO, w_hi, xtol=1e-12)
    grid = np.linspace(0.01, w_hi, 400001)
    phase = np.unwrap(np.angle(loop(grid)))
    return float(wc), float(180.0 + np.degrees(np.interp(wc, grid, phase)))


def margin_zoh(T):
    return crossover_and_margin(lambda w: loop_zoh(w, T), W_HI)


def margin_delay(T):
    return crossover_and_margin(lambda w: loop_delay(w, T), W_HI)


def margin_disc(T):
    return crossover_and_margin(lambda w: loop_disc(w, T), 0.999 * np.pi / T)


if __name__ == "__main__":
    wc0, pm0 = crossover_and_margin(loop_base, W_HI)
    print(f"연속 시간 개루프: 교차 각주파수 wc = {wc0:.2f} rad/s, 위상 여유 {pm0:.1f}도")
    print("ZOH의 위상과 크기 (T = 20 ms)")
    for w in (10.0, 24.0, 50.0, 100.0):
        h = zoh(w, 0.020)
        print(f"w={w:.0f} rad/s: 위상 {np.degrees(np.angle(h)):.2f}도 | -wT/2 {-np.degrees(w * 0.020 / 2):.2f}도 | 크기 {abs(h):.4f}")
    print("연속 시간 개루프에 ZOH를 곱했을 때와 길이 T/2인 지연을 곱했을 때, 이산 루프(지연 없음)의 위상 여유 [도]")
    for t_ms in (5, 10, 20, 40):
        T = t_ms / 1000
        wz, pz = margin_zoh(T)
        wd, pd = margin_delay(T / 2)
        wi, pi = margin_disc(T)
        print(f"T={t_ms} ms: ZOH {pz:.1f} (줄어든 각도 {pm0 - pz:.1f}) | 지연 T/2 {pd:.1f} | 이산 {pi:.1f} (줄어든 각도 {pm0 - pi:.1f}) | wc*T/2 {np.degrees(wc0 * T / 2):.1f}")

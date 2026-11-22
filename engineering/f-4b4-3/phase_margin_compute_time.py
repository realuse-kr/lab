"""F-4B4.3: 제어기 계산 시간 tc가 개루프 위상 여유를 얼마나 줄이는지 계산한다.

플랜트 P(s) = 1/(TAU s + 1), PI 제어기 C(s) = KP + KI/s, 개루프 L(s) = C(s) P(s).
계산 시간 tc는 샘플을 읽은 시각부터 출력이 바뀌는 시각까지 걸린 시간이고 0 <= tc <= T다.
ZOH의 주파수 응답은 (1 - e^(-jwT))/(jw)이고, 아래에서는 크기를 T로 나눠 (1 - e^(-jwT))/(jwT)를 쓴다.
세 루프를 비교한다.
 - 계산 시간 루프: 연속 시간 개루프에 ZOH 응답과 길이 tc인 지연 e^(-jw tc)를 곱한 루프.
 - 총 지연 루프: 연속 시간 개루프에 길이 T/2 + tc인 지연 하나를 곱한 루프.
 - 이산 루프: 출력이 kT + tc에 바뀌는 플랜트를 샘플 시각에서 정확히 이산화하고 제어기는 Tustin으로 이산화한 루프.
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
    """ZOH와 지연이 없는 연속 시간 개루프 L(jw)."""
    s = 1j * w
    return (KP + KI / s) / (TAU * s + 1)


def loop_compute(w, T, tc):
    """계산 시간 루프: 연속 시간 개루프에 ZOH 응답과 길이 tc인 지연을 곱한 값."""
    return loop_base(w) * zoh(w, T) * np.exp(-1j * w * tc)


def loop_total(w, T, tc):
    """총 지연 루프: 연속 시간 개루프에 길이 T/2 + tc인 지연 하나를 곱한 값."""
    return loop_base(w) * np.exp(-1j * w * (T / 2 + tc))


def loop_disc(w, T, tc):
    """이산 루프 L_d(e^(jwT)).

    출력이 kT + tc에 바뀌면 샘플 시각의 플랜트는 P(z) = (b0 + b1/z)/(z - alpha)다.
    alpha = exp(-T/TAU), b0 = 1 - exp(-(T-tc)/TAU), b1 = exp(-(T-tc)/TAU)(1 - exp(-tc/TAU)).
    tc = 0이면 b1 = 0이고 tc = T이면 b0 = 0이라 P(z)에 한 샘플 지연 1/z가 곱해진 꼴이다.
    """
    z = np.exp(1j * w * T)
    al = np.exp(-T / TAU)
    b0 = 1 - np.exp(-(T - tc) / TAU)
    b1 = np.exp(-(T - tc) / TAU) * (1 - np.exp(-tc / TAU))
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    return (c0 * z + c1) / (z - 1) * (b0 + b1 / z) / (z - al)


def crossover_and_margin(loop, w_hi):
    """개루프 크기가 1이 되는 각주파수 wc[rad/s]와 위상 여유[도].

    위상은 저주파에서 고주파로 촘촘한 격자를 따라 이어 붙인(unwrap) 값을 쓰고 wc에서 보간한다.
    """
    assert abs(loop(W_LO)) > 1.0 > abs(loop(w_hi)), "교차점이 구간 밖이다"
    wc = brentq(lambda w: abs(loop(w)) - 1.0, W_LO, w_hi, xtol=1e-12)
    grid = np.linspace(0.01, w_hi, 400001)
    phase = np.unwrap(np.angle(loop(grid)))
    return float(wc), float(180.0 + np.degrees(np.interp(wc, grid, phase)))


def margin_compute(T, tc):
    return crossover_and_margin(lambda w: loop_compute(w, T, tc), W_HI)


def margin_total(T, tc):
    return crossover_and_margin(lambda w: loop_total(w, T, tc), W_HI)


def margin_disc(T, tc):
    return crossover_and_margin(lambda w: loop_disc(w, T, tc), 0.999 * np.pi / T)


if __name__ == "__main__":
    wc0, pm0 = crossover_and_margin(loop_base, W_HI)
    print(f"연속 시간 개루프: 교차 각주파수 wc = {wc0:.2f} rad/s, 위상 여유 {pm0:.1f}도")
    T = 0.020
    print(f"샘플링 주기 T = 20 ms에서 계산 시간 tc를 바꾸었을 때 위상 여유 [도] (줄어든 각도는 {pm0:.1f}도에서 뺀 값)")
    for tc_ms in (0, 5, 10, 15, 20):
        tc = tc_ms / 1000
        _, pc = margin_compute(T, tc)
        _, pt = margin_total(T, tc)
        _, pd = margin_disc(T, tc)
        print(f"tc={tc_ms:2d} ms: 계산 시간 루프 {pc:.1f} (줄어든 각도 {pm0 - pc:.1f}) | 총 지연 루프 {pt:.1f} | 이산 루프 {pd:.1f} (줄어든 각도 {pm0 - pd:.1f}) | wc*(T/2+tc) {np.degrees(wc0 * (T / 2 + tc)):.1f}")

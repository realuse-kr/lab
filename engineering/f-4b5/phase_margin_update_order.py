"""F-4B5: 제어기가 출력을 내보내는 순서가 샘플에서 출력까지의 지연과 위상 여유를 얼마나 바꾸는지 계산한다.

플랜트 P(s) = 1/(TAU s + 1), PI 제어기 C(s) = KP + KI/s, 개루프 L(s) = C(s) P(s).
제어기는 샘플링 주기 T마다 샘플 읽기, 출력 계산, 출력 갱신을 한 번씩 한다. 계산에는 tc가 걸리고 0 <= tc <= T다.
출력 순서 두 가지를 비교한다.
 - 계산 직후 출력: 주기 안에서 읽고 계산한 뒤 바로 출력을 갱신한다. 샘플에서 출력까지 tc가 걸린다.
 - 다음 주기 시작에 출력: 주기 시작에 지난 주기에 계산한 출력을 갱신하고 그다음 읽고 계산한다. 샘플에서 출력까지 T가 걸린다.
출력 갱신 뒤에는 출력이 다음 갱신까지 일정하게 유지된다(ZOH).
위상 여유는 개루프 크기가 1이 되는 각주파수에서 개루프 위상에 180도를 더한 값이다.
"""
import numpy as np
from scipy.optimize import brentq

KP, KI, TAU = 2.0, 40.0, 0.1
W_LO, W_HI = 1.0, 200.0  # 개루프 크기가 1이 되는 각주파수를 찾는 구간 [rad/s]. KP, KI, TAU를 바꾸면 이 구간 안에 교차점이 있어야 한다.
ORDERS = ("compute_then_write", "write_then_compute")  # 계산 직후 출력, 다음 주기 시작에 출력


def schedule(order, T, tc, periods=4):
    """주기마다 (샘플 시각, 출력 갱신 시각) 쌍을 순서대로 만든다. 읽기와 출력 갱신은 시간이 걸리지 않고 계산만 tc가 걸린다."""
    records = []
    pending = None  # 지난 주기에 계산을 끝냈지만 아직 내보내지 않은 출력의 샘플 시각
    for k in range(periods):
        t = k * T
        if order == "compute_then_write":
            sample = t
            t += tc  # 계산
            records.append((sample, t))  # 계산이 끝난 시각에 출력 갱신
        elif order == "write_then_compute":
            if pending is not None:
                records.append((pending, t))  # 주기 시작에 지난 출력을 갱신
            sample = t
            t += tc  # 계산
            pending = sample
        else:
            raise ValueError(order)
    return records


def output_delay(order, T, tc):
    """schedule이 만든 쌍에서 샘플 시각과 출력 갱신 시각의 차이 [s]. 모든 쌍에서 같아야 한다."""
    delays = [round(out - smp, 12) for smp, out in schedule(order, T, tc)]
    assert max(delays) - min(delays) < 1e-9, delays
    return delays[0]


def zoh_phase_delay(T):
    """ZOH의 위상 지연에 해당하는 시간 T/2 [s]."""
    return T / 2


def loop_base(w):
    """지연이 없는 연속 시간 개루프 L(jw)."""
    s = 1j * w
    return (KP + KI / s) / (TAU * s + 1)


def loop_total(w, T, delay):
    """총 지연 루프: 연속 시간 개루프에 길이 T/2 + delay인 지연 하나를 곱한 값."""
    return loop_base(w) * np.exp(-1j * w * (T / 2 + delay))


def loop_disc(w, T, delay):
    """이산 루프 L_d(e^(jwT)). 출력이 샘플 시각에서 delay만큼 뒤(0 <= delay <= T)에 바뀌는 플랜트를 샘플 시각에서 정확히 이산화하고 제어기는 Tustin으로 이산화한다.

    P(z) = (b0 + b1/z)/(z - alpha), alpha = exp(-T/TAU), b0 = 1 - exp(-(T-delay)/TAU), b1 = exp(-(T-delay)/TAU)(1 - exp(-delay/TAU)).
    """
    z = np.exp(1j * w * T)
    al = np.exp(-T / TAU)
    b0 = 1 - np.exp(-(T - delay) / TAU)
    b1 = np.exp(-(T - delay) / TAU) * (1 - np.exp(-delay / TAU))
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    return (c0 * z + c1) / (z - 1) * (b0 + b1 / z) / (z - al)


def loop_disc_explicit(w, T, order, tc):
    """출력 순서를 식으로 직접 쓴 이산 루프. 계산 직후 출력이면 delay = tc, 다음 주기 시작에 출력이면 계산 시간과 상관없이 샘플 하나(1/z)를 곱한다.

    플랜트는 출력이 샘플 시각에 바로 바뀌는 경우의 P0(z) = (1 - alpha)/(z - alpha)로 쓰고 loop_disc의 계수 식을 쓰지 않는다.
    """
    z = np.exp(1j * w * T)
    al = np.exp(-T / TAU)
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2
    cd = (c0 * z + c1) / (z - 1)
    if order == "write_then_compute":
        return cd * (1 - al) / (z - al) / z
    raise ValueError("계산 직후 출력은 loop_disc를 쓴다")


def crossover_and_margin(loop, w_hi):
    """개루프 크기가 1이 되는 각주파수 wc[rad/s]와 위상 여유[도].

    위상은 저주파에서 고주파로 촘촘한 격자를 따라 이어 붙인(unwrap) 값을 쓰고 wc에서 보간한다.
    """
    assert abs(loop(W_LO)) > 1.0 > abs(loop(w_hi)), "교차점이 구간 밖이다"
    wc = brentq(lambda w: abs(loop(w)) - 1.0, W_LO, w_hi, xtol=1e-12)
    grid = np.linspace(0.01, w_hi, 400001)
    phase = np.unwrap(np.angle(loop(grid)))
    return float(wc), float(180.0 + np.degrees(np.interp(wc, grid, phase)))


def margin_disc(T, order, tc):
    d = output_delay(order, T, tc)
    return crossover_and_margin(lambda w: loop_disc(w, T, d), 0.999 * np.pi / T)


def margin_total(T, order, tc):
    d = output_delay(order, T, tc)
    return crossover_and_margin(lambda w: loop_total(w, T, d), W_HI)


if __name__ == "__main__":
    wc0, pm0 = crossover_and_margin(loop_base, W_HI)
    print(f"연속 시간 개루프: 교차 각주파수 wc = {wc0:.2f} rad/s, 위상 여유 {pm0:.1f}도")
    T = 0.010
    print("샘플링 주기 T = 10 ms에서 계산 시간 tc를 바꾸었을 때 샘플에서 출력까지의 지연 [ms]와 이산 루프 위상 여유 [도]")
    for tc_ms in (0, 2, 4, 6, 8, 10):
        tc = tc_ms / 1000
        da, db = output_delay("compute_then_write", T, tc), output_delay("write_then_compute", T, tc)
        pa = margin_disc(T, "compute_then_write", tc)[1]
        pb = margin_disc(T, "write_then_compute", tc)[1]
        print(f"tc={tc_ms:2d} ms: 계산 직후 출력 지연 {da*1e3:4.1f} ms 위상 여유 {pa:.1f} | 다음 주기 시작에 출력 지연 {db*1e3:4.1f} ms 위상 여유 {pb:.1f} | 차이 {pa - pb:.1f} | wc*(T-tc) {np.degrees(wc0 * (T - tc)):.1f}")

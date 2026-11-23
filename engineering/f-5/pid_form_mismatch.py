"""F-5: 같은 PID 숫자를 병렬형과 이상형에 바꿔 넣었을 때의 스텝 응답.
플랜트 P(s) = 1 / ((0.1 s + 1)(0.02 s + 1)), 목표값 1인 스텝, 미분 필터 없음."""
import numpy as np
from scipy import signal

A = np.polymul([0.1, 1], [0.02, 1])  # 플랜트 분모


def ideal_to_parallel(kc, ti, td):
    """이상형 (Kc, Ti, Td) -> 병렬형 (Kp, Ki, Kd)"""
    return kc, kc / ti, kc * td


def parallel_to_ideal(kp, ki, kd):
    """병렬형 (Kp, Ki, Kd) -> 이상형 (Kc, Ti, Td)"""
    return kp, kp / ki, kd / kp


def closed_loop(kp, ki, kd):
    """폐루프 T(s) = N/D. C(s) = (Kd s^2 + Kp s + Ki)/s, N = Kd s^2 + Kp s + Ki, D = s A(s) + N"""
    num = np.array([kd, kp, ki])
    return num, np.polyadd(np.polymul([1, 0], A), num)


def step_response(kp, ki, kd, t):
    """T(s)/s를 부분분수(residue)로 전개해 y(t)를 구한다."""
    num, den = closed_loop(kp, ki, kd)
    r, p, _ = signal.residue(num, np.polymul(den, [1, 0]))
    return np.real(sum(ri * np.exp(pi * t) for ri, pi in zip(r, p)))


def metrics(t, y, band=0.02):
    """오버슈트 [%], 정정 시간 [s](±2 % 밖에 마지막으로 있던 시각의 다음 격자점)"""
    outside = np.where(np.abs(y - 1.0) > band)[0]
    ts = t[min(outside[-1] + 1, len(t) - 1)] if len(outside) else 0.0
    return 100 * max(y.max() - 1.0, 0.0), ts


if __name__ == "__main__":
    t = np.linspace(0, 1.0, 100001)
    cases = {  # 이름: 실제로 적용되는 (Kp, Ki, Kd)
        "이상형 Kc 2, Ti 0.1, Td 0.02를 이상형으로 옮김": ideal_to_parallel(2.0, 0.1, 0.02),
        "같은 숫자를 병렬형 Kp, Ki, Kd에 그대로": (2.0, 0.1, 0.02),
        "병렬형 숫자 2, 20, 0.04를 이상형 Kc, Ti, Td에 그대로": ideal_to_parallel(2.0, 20.0, 0.04),
    }
    for name, (kp, ki, kd) in cases.items():
        y = step_response(kp, ki, kd, t)
        print(f"{name}: Ki {ki:g}, Kd {kd:g}, y(0.6 s) {y[60000]:.3f}, 오버슈트 {metrics(t, y)[0]:.1f} %")

import numpy as np
from scipy.signal import cont2discrete, step

TAU, KP, KI = 0.1, 2.0, 40.0  # 플랜트 시간상수 [s], PI 제어기의 Kp와 Ki
t = np.linspace(0, 1.5, 150001)  # 기준 응답의 시각 [s], 간격 10 us
_, yc = step(([KP / TAU, KI / TAU], [1, (1 + KP) / TAU, KI / TAU]), T=t)  # 연속 시간 폐루프
OS_CONT = (yc.max() - 1) * 100  # 연속 시간 오버슈트 [%]
METHODS = {"Forward Euler": "euler", "Backward Euler": "backward_diff", "Tustin": "bilinear"}

def run(method, T, tend=1.5):  # 폐루프 스텝 응답의 오버슈트 [%]와 최대 |z|
    num, den, _ = cont2discrete(([KP, KI], [1, 0]), T, method=method)
    b0, b1 = np.squeeze(num)  # C(z) = (b0 z + b1) / (z - 1)
    a = np.exp(-T / TAU)  # 영차 유지 플랜트: y[k+1] = a y[k] + (1 - a) u[k]
    p = np.polyadd(np.polymul([1, -a], [1, -1]), (1 - a) * np.array([b0, b1]))  # 폐루프 특성식
    y = u = e_old = peak = 0.0
    for _ in range(round(tend / T)):
        e = 1.0 - y
        u += b0 * e + b1 * e_old  # u[k] = u[k-1] + b0 e[k] + b1 e[k-1]
        y, e_old = a * y + (1 - a) * u, e
        peak = max(peak, y)  # 샘플 사이에서 y는 단조라 최댓값은 샘플 시각에 있다
    return (peak - 1) * 100, np.abs(np.roots(p)).max()

if __name__ == "__main__":
    print(f"연속 시간 오버슈트 {OS_CONT:.2f}%")
    for T in (0.005, 0.01, 0.02, 0.04, 0.05):
        r = {n: run(k, T) for n, k in METHODS.items()}
        print(f"T={T * 1e3:g}ms | " + " | ".join(
            f"{n} {o:.2f}% ({o - OS_CONT:+.2f}%p, 최대|z| {z:.3f})" for n, (o, z) in r.items()))

import numpy as np
from scipy.signal import cont2discrete

KP, KI = 2.0, 40.0  # PI 제어기의 Kp와 Ki
METHODS = {"Forward Euler": "euler", "Backward Euler": "backward_diff", "Tustin": "bilinear"}

def mag_error(method, T, w):  # 크기 오차 [%] = (|C(e^jwT)| / |C(jw)| - 1) x 100
    num, den, _ = cont2discrete(([KP, KI], [1, 0]), T, method=method)
    b, a = np.squeeze(num), np.squeeze(den)  # C(z) = (b0 z + b1) / (z - 1)
    z = np.exp(1j * w * T)  # 주파수 w [rad/s]의 사인파에 해당하는 z
    Cd = np.polyval(b, z) / np.polyval(a, z)
    Cc = KP + KI / (1j * w)  # 연속 시간 제어기 C(jw)
    return (np.abs(Cd) / np.abs(Cc) - 1) * 100, b

if __name__ == "__main__":
    T = 0.02
    w = np.geomspace(0.1, np.pi / T, 400001)  # 0.1 rad/s ~ 나이퀴스트 주파수 pi/T
    print(f"T={T * 1e3:g}ms, 나이퀴스트 주파수 {np.pi / T:.2f} rad/s")
    for name, m in METHODS.items():
        e, b = mag_error(m, T, w)
        i = np.argmax(np.abs(e))
        over = np.abs(e) > 1
        first = f"{w[np.argmax(over)]:.2f} rad/s" if over.any() else "없음"
        print(f"{name}: b0={b[0]:.2f}, b1={b[1]:.2f} | 최대 크기 오차 {e[i]:+.2f}% "
              f"({w[i]:.1f} rad/s) | 1% 처음 넘는 주파수 {first}")

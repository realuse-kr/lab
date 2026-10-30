import numpy as np
from scipy.signal import cont2discrete, lfilter

TAU = 0.1  # 플랜트 P(s) = 1 / (TAU*s + 1)의 시간 상수 [s]
METHODS = {"ZOH": "zoh", "Forward Euler": "euler", "Backward Euler": "backward_diff", "Tustin": "bilinear"}

def discretize(method, T):  # P(z) = (b0 z + b1) / (z - a)
    num, den, _ = cont2discrete(([1.0], [TAU, 1.0]), T, method=method)
    return np.squeeze(num), np.squeeze(den)

def step_error(method, T, t_end=2.0):  # 샘플 시점의 스텝 응답 오차 [% of 최종값 1]
    num, den = discretize(method, T)
    k = np.arange(int(round(t_end / T)) + 1)
    y = lfilter(num, den, np.ones(k.size))  # 입력이 매 샘플 1로 유지되는 스텝
    return (y - (1 - np.exp(-k * T / TAU))) * 100  # 연속 시간 해석해 1 - exp(-t/TAU)와의 차

if __name__ == "__main__":
    T = 0.02
    print(f"TAU={TAU} s, T={T * 1e3:g} ms, exp(-T/TAU)={np.exp(-T / TAU):.4f}")
    for name, m in METHODS.items():
        num, den = discretize(m, T)
        e = step_error(m, T)
        i = np.argmax(np.abs(e))
        print(f"{name}: 극점 z={-den[1]:.4f} | 최대 응답 오차 {e[i]:+.3e}% ({i * T * 1e3:g} ms)")

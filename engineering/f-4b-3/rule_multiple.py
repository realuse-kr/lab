import numpy as np
from scipy import signal
KP, KI, TAU, TEND = 2.0, 40.0, 0.1, 2.0 # 플랜트 P(s) = 1/(TAU*s + 1), PI 제어기 C(s) = KP + KI/s, 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI] # 연속 시간 폐루프 전달함수 (KP*s + KI)/(TAU*s^2 + (1 + KP)*s + KI)
def bandwidth_hz(): # 폐루프 크기가 처음 1/sqrt(2)보다 작아지는 주파수(-3 dB 대역폭)
    w = np.logspace(0, 3, 200000)
    return w[np.argmax(np.abs(signal.freqs(NUM, DEN, w)[1]) < 1 / np.sqrt(2))] / (2 * np.pi)
def rules(T): # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1)
    return {"Forward Euler": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP), "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}
def deviation(T): # 규칙별로 샘플 시점마다 이산 응답과 연속 응답의 차를 구해 처음 TEND초 동안 가장 큰 절댓값
    n, a, out = int(round(TEND / T)), np.exp(-T / TAU), {}
    yc = signal.step((NUM, DEN), T=np.arange(n + 1) * T)[1] # 연속 시간 폐루프의 스텝 응답
    for rule, (c0, c1) in rules(T).items():
        y, e_old, u_old = [0.0], 0.0, 0.0
        for _ in range(n):
            e = 1.0 - y[-1]
            u = c0 * e + c1 * e_old + u_old
            y.append(a * y[-1] + (1 - a) * u) # ZOH 플랜트 y[k+1] = a*y[k] + (1 - a)*u[k]
            e_old, u_old = e, u
        out[rule] = np.abs(np.array(y) - yc).max()
    return out
def phase_margin(T, rule): # 이산 개루프 L(z) = (c0 z + c1)/(z - 1) * (1 - a)/(z - a)의 크기가 처음 1보다 작아지는 점에서 180 + 위상(도)
    a, (c0, c1) = np.exp(-T / TAU), rules(T)[rule]
    th = np.linspace(1e-3, np.pi * 0.9999, 200000) # th = 각주파수 * T
    L = (c0 * np.exp(1j * th) + c1) / (np.exp(1j * th) - 1) * (1 - a) / (np.exp(1j * th) - a)
    return 180 + np.degrees(np.unwrap(np.angle(L))[np.argmax(np.abs(L) < 1)])
if __name__ == "__main__":
    for m in (6, 8, 10, 20, 25, 30, 40, 100):
        T = 1 / (m * bandwidth_hz()) # 샘플링 비율 m = 샘플링 주파수 1/T를 대역폭으로 나눈 값
        print(f"m={m:>3} (T={1000 * T:5.1f} ms) 편차 " + " | ".join(f"{k} {v:.3f}" for k, v in deviation(T).items()) + " | 위상 여유 " + " | ".join(f"{phase_margin(T, k):.1f}" for k in rules(T)))

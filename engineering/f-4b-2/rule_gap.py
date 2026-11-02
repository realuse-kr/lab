import numpy as np
from scipy import signal
KP, KI, TAU, TEND = 2.0, 40.0, 0.1, 2.0 # 플랜트 P(s) = 1/(TAU*s + 1), PI 제어기 C(s) = KP + KI/s, 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI] # 연속 시간 폐루프 전달함수 (KP*s + KI)/(TAU*s^2 + (1 + KP)*s + KI)
def bandwidth_hz(): # 폐루프 크기가 처음 1/sqrt(2)보다 작아지는 주파수(-3 dB 대역폭)
    w = np.logspace(0, 3, 200000)
    return w[np.argmax(np.abs(signal.freqs(NUM, DEN, w)[1]) < 1 / np.sqrt(2))] / (2 * np.pi)
def rules(T): # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1)
    return {"Forward Euler": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP), "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}
def responses(T): # 규칙별 샘플 시점 응답 y[0..n], 플랜트는 ZOH 모델 y[k+1] = a*y[k] + (1 - a)*u[k]
    n, a, out = int(round(TEND / T)), np.exp(-T / TAU), {}
    for rule, (c0, c1) in rules(T).items():
        y, e_old, u_old = [0.0], 0.0, 0.0
        for _ in range(n):
            e = 1.0 - y[-1]
            u = c0 * e + c1 * e_old + u_old
            y.append(a * y[-1] + (1 - a) * u)
            e_old, u_old = e, u
        out[rule] = np.array(y)
    return out
def max_gap(T): # 규칙 간 최대 차이 = 샘플 시점마다 (세 규칙 응답의 최댓값 - 최솟값)을 구해 그중 가장 큰 값
    return np.ptp(np.array(list(responses(T).values())), axis=0).max()
def pole_modulus(T, rule): # 폐루프 특성다항식 (z - 1)(z - a) + (1 - a)(c0 z + c1)의 근 중 가장 큰 크기
    a, (c0, c1) = np.exp(-T / TAU), rules(T)[rule]
    return max(abs(np.roots(np.polyadd(np.polymul([1, -1], [1, -a]), (1 - a) * np.array([c0, c1])))))
if __name__ == "__main__":
    fb = bandwidth_hz() # 샘플링 비율 r = 샘플링 주파수 1/T를 대역폭 fb로 나눈 값
    for r in (3, 5, 10, 20, 40, 80, 160):
        T = 1 / (r * fb)
        print(f"r={r:>3} (T={1000 / (r * fb):5.1f} ms) 규칙 간 최대 차이 {max_gap(T):.3f} | 극점 최대 크기 " + " | ".join(f"{k} {pole_modulus(T, k):.3f}" for k in rules(T)))

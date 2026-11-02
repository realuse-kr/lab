import numpy as np
from scipy import signal
KP, KI, TAU, TEND = 2.0, 40.0, 0.1, 2.0 # 플랜트 P(s) = 1/(TAU*s + 1), PI 제어기 C(s) = KP + KI/s, 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI] # 연속 시간 폐루프 전달함수 (KP*s + KI)/(TAU*s^2 + (1 + KP)*s + KI)
def bandwidth_hz(): # 폐루프 크기가 처음 1/sqrt(2)보다 작아지는 주파수(-3 dB 대역폭)
    w = np.logspace(0, 3, 200000)
    return w[np.argmax(np.abs(signal.freqs(NUM, DEN, w)[1]) < 1 / np.sqrt(2))] / (2 * np.pi)
def rules(T): # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1)
    return {"Forward Euler": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP), "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}
def metrics(T): # 규칙별 (오버슈트, 연속 시간 응답과의 최대 편차)와 연속 시간 오버슈트
    n, a = int(round(TEND / T)), np.exp(-T / TAU) # 플랜트는 ZOH 모델 y[k+1] = a*y[k] + (1 - a)*u[k]
    yc = signal.step((NUM, DEN), T=np.arange(n + 1) * T)[1] # 연속 시간 폐루프의 같은 시각 값
    out = {}
    for rule, (c0, c1) in rules(T).items():
        y, e_old, u_old = [0.0], 0.0, 0.0
        for _ in range(n):
            e = 1.0 - y[-1]
            u = c0 * e + c1 * e_old + u_old
            y.append(a * y[-1] + (1 - a) * u)
            e_old, u_old = e, u
        out[rule] = (max(y) - 1.0, np.max(np.abs(np.array(y) - yc)))
    return out, signal.step((NUM, DEN), T=np.linspace(0, TEND, 20001))[1].max() - 1.0 # 연속 시간 오버슈트는 촘촘한 시각에서 잰다
if __name__ == "__main__":
    fb = bandwidth_hz() # 샘플링 비율 r = 샘플링 주파수 1/T를 대역폭 fb로 나눈 값
    for r in (5, 10, 20, 40, 80):
        out, oc = metrics(1 / (r * fb))
        print(f"r={r:>2} (T={1000 / (r * fb):4.1f} ms) 연속 {oc * 100:.1f}% | " + " | ".join(f"{k} {v[0] * 100:4.1f}% {v[1]:.3f}" for k, v in out.items()))

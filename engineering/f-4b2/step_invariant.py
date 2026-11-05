import numpy as np
from scipy.signal import cont2discrete
KP, KI = 2.0, 40.0 # PI 제어기 C(s) = KP + KI/s (입력은 오차 e, 출력은 제어기 출력 u)
def step_invariant(T): # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1): 스텝 불변(ZOH) 이산화
    num, den, _ = cont2discrete(([KP, KI], [1.0, 0.0]), T, method="zoh")
    assert np.allclose(np.squeeze(den), [1.0, -1.0]) # 분모는 z - 1
    return tuple(np.squeeze(num))
def rules(T):
    return {"스텝 불변": step_invariant(T), "Forward Euler": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP),
            "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}
def controller_out(c, e): # u[k] = c0 e[k] + c1 e[k-1] + u[k-1]
    u, e_old, u_old = [], 0.0, 0.0
    for ek in e:
        u_old = c[0] * ek + c[1] * e_old + u_old
        e_old = ek
        u.append(u_old)
    return np.array(u)
if __name__ == "__main__":
    T = 0.02
    t = np.arange(101) * T # 처음 2초의 샘플 시점
    for name, c in rules(T).items():
        step = controller_out(c, np.ones(t.size)) - (KP + KI * t) # 스텝 입력 e=1: 연속 시간 u(t) = KP + KI*t
        ramp = controller_out(c, t) - (KP * t + KI * t ** 2 / 2) # 램프 입력 e=t: 연속 시간 u(t) = KP*t + KI*t^2/2
        print(f"{name:>14}: c0={c[0]:.4f} c1={c[1]:+.4f} | 스텝 입력 차이 {step.min():+.4f}~{step.max():+.4f} | 램프 입력 차이(2초) {ramp[-1]:+.4f}")

import numpy as np
from scipy.signal import zpk2tf
KP, KI = 2.0, 40.0                # PI 제어기 C(s) = KP + KI/s = KP (s + A)/s, 영점 s = -A
A = KI / KP                       # 영점 -A = -20 rad/s, 극점은 s = 0
def matched(T):                   # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1): 극·영점 매칭
    p = np.exp(-A * T)            # 영점 s=-A -> z=exp(-A T), 극점 s=0 -> z=1
    k = KI * T / (1 - p)          # 이득: 낮은 주파수에서 (z-1)C(z)/T가 KI가 되도록 정한다
    num, den = zpk2tf([p], [1.0], k)
    assert np.allclose(den, [1.0, -1.0])
    return tuple(num)             # (k, -k p)
def rules(T):
    return {"극·영점 매칭": matched(T), "스텝 불변": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP),
            "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}
def controller_out(c, e):         # u[k] = c0 e[k] + c1 e[k-1] + u[k-1]
    u, e_old, u_old = [], 0.0, 0.0
    for ek in e:
        u_old = c[0] * ek + c[1] * e_old + u_old
        e_old = ek
        u.append(u_old)
    return np.array(u)
if __name__ == "__main__":
    T = 0.02
    t = np.arange(101) * T        # 처음 2초의 샘플 시점
    for name, c in rules(T).items():
        step = controller_out(c, np.ones(t.size)) - (KP + KI * t)               # 스텝 입력 e=1
        ramp = controller_out(c, t) - (KP * t + KI * t ** 2 / 2)                # 램프 입력 e=t
        expo = controller_out(c, np.exp(-A * t)) - KP                           # 입력 e=exp(-A t): 연속 시간 출력은 KP로 일정
        print(f"{name:>12}: c0={c[0]:.4f} c1={c[1]:+.4f} 영점={-c[1] / c[0]:.4f} | 스텝 입력 차이 {step[-1]:+.4f} | 램프 입력 차이(2초) {ramp[-1]:+.4f} | exp 입력 차이 {expo.min():+.4f}~{expo.max():+.4f}")

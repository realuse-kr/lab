import numpy as np
KP, KI = 2.0, 40.0  # PI 제어기 C(s) = KP + KI/s
def tustin(T):  # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1): 프리워핑 없는 Tustin
    return (KP + KI * T / 2, -KP + KI * T / 2)
def warp_gain(T, w0):  # g = tan(w0 T/2)/(w0 T/2), w0 -> 0이면 1
    x = w0 * T / 2
    return np.tan(x) / x
def prewarped(T, w0):  # 프리워핑 Tustin: 적분 항의 T/2를 T g/2로 바꾼다
    g = warp_gain(T, w0)
    return (KP + KI * T * g / 2, -KP + KI * T * g / 2)
def substitute(T, w0):  # s = a (z-1)/(z+1), a = w0/tan(w0 T/2)를 C(s) = (KP s + KI)/s에 직접 대입해 (c0, c1)을 구한다
    a = w0 / np.tan(w0 * T / 2)
    num = np.array([KP * a + KI, -KP * a + KI]) / a  # 분자 계수, 분모 a(z-1)을 a로 나눠 (z-1)로 맞춘다
    return tuple(num)
def ratio(c, T, w):  # 이산 제어기와 연속 시간 제어기의 주파수 응답비 Cd(exp(jwT))/C(jw)
    z = np.exp(1j * w * T)
    return (c[0] * z + c[1]) / (z - 1) / (KP + KI / (1j * w))
def controller_out(c, e):  # u[k] = u[k-1] + c0 e[k] + c1 e[k-1]
    u, e_old, u_old = [], 0.0, 0.0
    for ek in e:
        u_old = c[0] * ek + c[1] * e_old + u_old
        e_old = ek
        u.append(u_old)
    return np.array(u)
if __name__ == "__main__":
    T, w0 = 0.02, 24.0
    c = prewarped(T, w0)
    assert np.allclose(c, substitute(T, w0))
    print(f"g={warp_gain(T, w0):.5f} c0={c[0]:.4f} c1={c[1]:+.4f} 영점={-c[1] / c[0]:.4f}")
    for name, cc in (("Tustin", tustin(T)), (f"프리워핑 {w0:g} rad/s", c)):
        r = ratio(cc, T, w0)
        print(f"{name:>16}: w0에서 크기비 {abs(r):.4f} 위상차 {np.degrees(np.angle(r)):+.3f}도")

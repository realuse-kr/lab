import numpy as np
KP, KI, T, TAU = 2.0, 40.0, 0.02, 0.1  # 플랜트 P(s) = 1/(TAU*s + 1), PI 제어기 C(s) = KP + KI/s, 주기 T
RULES = {"ZOH, Forward Euler": (KP, -KP + KI * T), "Backward Euler": (KP + KI * T, -KP), "Tustin": (KP + KI * T / 2, -KP + KI * T / 2)}  # C(z) = (c0 z + c1)/(z - 1)의 (c0, c1)
def step_response(rule, delay, n=60):  # delay = 제어기 출력이 플랜트에 들어가기 전에 거치는 샘플 수
    a, (c0, c1) = np.exp(-T / TAU), RULES[rule]
    y, buf, e_old, u_old = [0.0], [0.0] * delay, 0.0, 0.0
    for _ in range(n):
        e = 1.0 - y[-1]  # 오차 = 목표값 1 - 출력
        u = c0 * e + c1 * e_old + u_old  # 제어기 차분방정식
        buf.append(u)
        y.append(a * y[-1] + (1 - a) * buf.pop(0))  # 플랜트 ZOH 모델, 입력은 delay 샘플 전에 계산한 u
        e_old, u_old = e, u
    return np.array(y)
def read_poles(y, tol=1e-9):  # 스텝 응답에서 극점을 읽는다: 차수 n을 1부터 올려 처음 맞는 n이 극점 수
    h = np.diff(np.r_[0.0, y])  # 스텝 응답의 차분 = 충격 응답
    for n in range(1, 6):
        A = np.array([[-h[k - j] for j in range(1, n + 1)] for k in range(n + 3, len(h))])  # 처음 3개 샘플은 쓰지 않는다
        p = np.linalg.lstsq(A, h[n + 3:], rcond=None)[0]  # h[k] = -p1 h[k-1] - ... - pn h[k-n]
        if np.linalg.norm(A @ p - h[n + 3:]) < tol: return n, np.roots(np.r_[1.0, p])
def char_poles(rule, delay):  # 특성다항식 z^delay (z - 1)(z - a) + (1 - a)(c0 z + c1)의 근
    a, (c0, c1) = np.exp(-T / TAU), RULES[rule]
    return np.roots(np.polyadd(np.polymul(np.polymul([1, -1], [1, -a]), np.r_[1.0, np.zeros(delay)]), (1 - a) * np.array([c0, c1])))
if __name__ == "__main__":
    for rule in RULES:
        for delay in (0, 1):
            n, p = read_poles(step_response(rule, delay))
            gap = max(min(abs(p - r)) for r in char_poles(rule, delay))
            print(f"{rule}, 지연 {delay}: 극점 {n}개 {np.round(np.sort_complex(p), 4)} | 해석값과 차이 {gap:.0e}")

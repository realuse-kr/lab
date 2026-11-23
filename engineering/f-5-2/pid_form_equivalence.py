"""F-5.2: 병렬형 PID와 이상형 PID를 변환식으로 맞추면 스텝 응답이 같은지 확인한다.
플랜트 P(s) = 1 / ((0.1 s + 1)(0.02 s + 1)), 목표값 1인 스텝, 미분 필터 없음."""
import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp

A = np.polymul([0.1, 1], [0.02, 1])  # 플랜트 분모 a2 s^2 + a1 s + 1
A2, A1 = A[0], A[1]


def ideal_to_parallel(kc, ti, td):
    """이상형 (Kc, Ti, Td) -> 병렬형 (Kp, Ki, Kd)"""
    return kc, kc / ti, kc * td


def parallel_to_ideal(kp, ki, kd):
    """병렬형 (Kp, Ki, Kd) -> 이상형 (Kc, Ti, Td)"""
    return kp, kp / ki, kd / kp


def _step_from_tf(num, den, t):
    """폐루프 T(s) = num/den의 스텝 응답. T(s)/s를 부분분수로 전개한다."""
    r, p, _ = signal.residue(num, np.polymul(den, [1, 0]))
    return np.real(sum(ri * np.exp(pi * t) for ri, pi in zip(r, p)))


def step_tf_parallel(kp, ki, kd, t):
    """병렬형 C(s) = (Kd s^2 + Kp s + Ki) / s 로 세운 폐루프의 스텝 응답. 반환: (y, 폐루프 분모)"""
    num = np.array([kd, kp, ki])
    den = np.polyadd(np.polymul([1, 0], A), num)
    return _step_from_tf(num, den, t), den


def step_tf_ideal(kc, ti, td, t):
    """이상형 C(s) = Kc (Ti Td s^2 + Ti s + 1) / (Ti s) 로 세운 폐루프의 스텝 응답.
    변환식을 쓰지 않고 Kc, Ti, Td를 그대로 식에 넣는다. 반환: (y, 폐루프 분모)"""
    num = kc * np.array([ti * td, ti, 1.0])
    den = np.polyadd(np.polymul([ti, 0.0], A), num)
    return _step_from_tf(num, den, t), den


def step_ode(law, impulse_gain, t, rtol=1e-12, atol=1e-14):
    """폐루프를 시간 영역에서 직접 적분한다. 상태는 출력 y, 변화율 v, 오차의 적분 z다.
    law(e, z, de)는 PID 식 하나이고 e는 오차, de는 오차의 미분이다.
    목표값이 0에서 1로 뛰는 순간 미분 항이 임펄스를 내서 v가 impulse_gain / a2만큼 뛴다."""
    def f(_, x):
        y, v, z = x
        e = 1.0 - y
        u = law(e, z, -v)
        return [v, (u - A1 * v - y) / A2, e]
    sol = solve_ivp(f, (t[0], t[-1]), [0.0, impulse_gain / A2, 0.0], t_eval=t,
                    method="DOP853", rtol=rtol, atol=atol)
    return sol.y[0]


def step_ode_parallel(kp, ki, kd, t):
    return step_ode(lambda e, z, de: kp * e + ki * z + kd * de, kd, t)


def step_ode_ideal(kc, ti, td, t):
    return step_ode(lambda e, z, de: kc * (e + z / ti + td * de), kc * td, t)


def metrics(t, y, band=0.02):
    """오버슈트 [%], 정정 시간 [s](±2 % 밖에 마지막으로 있던 시각의 다음 격자점)"""
    outside = np.where(np.abs(y - 1.0) > band)[0]
    ts = t[min(outside[-1] + 1, len(t) - 1)] if len(outside) else 0.0
    return 100 * max(y.max() - 1.0, 0.0), ts


if __name__ == "__main__":
    t = np.linspace(0, 1.0, 100001)
    kc, ti, td = 2.0, 0.1, 0.02
    kp, ki, kd = ideal_to_parallel(kc, ti, td)
    y_par, _ = step_tf_parallel(kp, ki, kd, t)
    y_ide, _ = step_tf_ideal(kc, ti, td, t)
    y_wrong, _ = step_tf_parallel(kc, ti, td, t)  # 이상형 숫자를 병렬형에 그대로
    print(f"병렬형 Kp {kp:g}, Ki {ki:g}, Kd {kd:g}: y(0.6 s) {y_par[60000]:.4f}")
    print(f"이상형 Kc {kc:g}, Ti {ti:g}, Td {td:g}: y(0.6 s) {y_ide[60000]:.4f}")
    print(f"두 응답의 최대 차이 {np.abs(y_par - y_ide).max():.1e}")
    print(f"숫자를 그대로 넣은 응답과의 최대 차이 {np.abs(y_wrong - y_par).max():.3f}")

"""F-4B5.2: 출력 갱신 순서에 따라 PI 제어 루프의 스텝 응답이 얼마나 달라지는지 계산한다.

플랜트 P(s) = 1/(TAU s + 1), 제어기는 Tustin으로 이산화한 PI 제어기다.
제어기는 샘플링 주기 T마다 샘플 읽기, 출력 계산, 출력 갱신을 한 번씩 한다. 계산에는 tc가 걸리고 0 <= tc <= T다.
 - 계산 직후 출력: 샘플을 읽고 계산이 끝나면 바로 출력을 갱신한다. 샘플에서 출력까지 d = tc가 걸린다.
 - 다음 주기 시작에 출력: 계산 결과를 보관했다가 다음 주기가 시작될 때 출력을 갱신한다. 샘플에서 출력까지 d = T가 걸린다.
출력 갱신 뒤에는 다음 갱신까지 플랜트 입력이 일정하다(ZOH). 플랜트는 연속 시간으로 정확히 계산한다.
목표값은 크기 1인 스텝이고 오버슈트·정정 시간은 샘플 사이까지 포함한 연속 출력 y(t)에서 구한다.
"""
import numpy as np

KP, KI, TAU = 2.0, 40.0, 0.1
ORDERS = ("compute_then_write", "write_then_compute")  # 계산 직후 출력, 다음 주기 시작에 출력
BAND = 0.02  # 정정 시간 기준: 최종값의 ±2 %


def hold(x0, u, length):
    """입력 u를 length 동안 일정하게 넣었을 때 1차 플랜트의 상태. 정확한 해다."""
    return u + (x0 - u) * np.exp(-length / TAU)


def step_response(order, T, tc, t_end=0.6, sub=100):
    """목표값 1인 스텝 응답. 반환값은 시간 t [s], 연속 출력 y(t), 계산한 제어 입력 u_k 목록이다. y[::sub]는 샘플 시각의 출력이다."""
    c0, c1 = KP + KI * T / 2, -KP + KI * T / 2            # Tustin PI: u_k = u_(k-1) + c0 e_k + c1 e_(k-1)
    d = tc if order == "compute_then_write" else T       # 샘플에서 출력까지의 시간
    x = u_prev = e_prev = 0.0
    y, us = [], []
    for k in range(int(round(t_end / T))):
        e = 1.0 - x                                      # 샘플 읽기
        u = u_prev + c0 * e + c1 * e_prev                # 출력 계산
        for j in range(sub):                             # 이 주기의 y(t)를 sub칸으로 기록
            s = j * T / sub
            y.append(hold(x, u_prev, s) if s < d else hold(hold(x, u_prev, d), u, s - d))
        x = hold(hold(x, u_prev, d), u, T - d)           # d까지는 지난 입력, 그 뒤에는 새 입력
        u_prev, e_prev = u, e
        us.append(u)
    y.append(x)
    return np.arange(len(y)) * T / sub, np.array(y), np.array(us)


def metrics(t, y):
    """오버슈트 [%], 정점 시각 [s], 정정 시간 [s](y가 ±2 % 밖에 마지막으로 있던 시각 다음 격자점)."""
    ip = int(np.argmax(y))
    outside = np.where(np.abs(y - 1.0) > BAND)[0]
    settle = float(t[outside[-1] + 1]) if len(outside) else 0.0
    return 100.0 * (float(y[ip]) - 1.0), float(t[ip]), settle


def continuous_reference(t_end=0.6, dt=1e-4):
    """샘플링과 지연이 없는 연속 시간 폐루프 Y/R = (KP s + KI)/(TAU s^2 + (1 + KP) s + KI)의 스텝 응답."""
    from scipy.signal import TransferFunction, step
    t = np.arange(0.0, t_end + dt / 2, dt)
    return step(TransferFunction([KP, KI], [TAU, 1 + KP, KI]), T=t)


if __name__ == "__main__":
    t0, y0 = continuous_reference()
    o0, p0, s0 = metrics(t0, y0)
    print(f"연속 시간 폐루프(샘플링 없음): 오버슈트 {o0:.1f} %, 정점 {p0 * 1e3:.0f} ms, 정정 시간 {s0 * 1e3:.0f} ms")
    T = 0.010
    print("샘플링 주기 T = 10 ms에서 계산 시간 tc를 바꾸었을 때 오버슈트 [%] / 정점 시각 [ms] / 정정 시간 [ms]")
    for tc_ms in (0, 2, 4, 6, 8, 10):
        row = []
        for order in ORDERS:
            t, y, _ = step_response(order, T, tc_ms / 1000)
            o, p, s = metrics(t, y)
            row.append(f"{o:5.1f} / {p * 1e3:3.0f} / {s * 1e3:3.0f}")
        print(f"tc={tc_ms:2d} ms: 계산 직후 출력 {row[0]} | 다음 주기 시작에 출력 {row[1]}")

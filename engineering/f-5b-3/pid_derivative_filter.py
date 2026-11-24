"""F-5B.3: PID 미분 항에 1차 필터를 달면 설정값 계단의 미분 킥이 어떻게 달라지는지 재 보기.

PID는 병렬형이다: u = Kp*e + Ki*(e의 적분) + d. e = r - y (r 설정값, y 측정값).
미분 필터는 연속 시간식 Tf*(d의 변화 속도) + d = Kd*(e의 변화 속도)이고,
변화 속도를 backward difference로 바꾸면 다음 점화식이 된다.
  d[k] = Tf/(Tf+Ts) * d[k-1] + Kd/(Tf+Ts) * (e[k] - e[k-1])
Tf = 0이면 필터가 없는 backward difference Kd*(e[k]-e[k-1])/Ts와 같다.
제어 대상은 1차 지연 플랜트 tau*y' = -y + K*u이고, 제어 출력은 한 샘플 동안 일정하게 유지한다.
"""
import numpy as np

KP = 2.0       # 비례 게인
KI = 2.0       # 적분 게인
KD = 0.04      # 미분 게인 (Td = KD / KP = 0.02 s)
TAU = 0.5      # 플랜트 시간상수 [s]
K_PLANT = 1.0  # 플랜트 정상상태 이득
TS = 1e-3      # 샘플링 주기 [s]
DR = 1.0       # 설정값 계단의 크기
T_STEP = 0.1   # 설정값이 바뀌는 시각 [s]
T_END = 1.0    # 시뮬레이션 길이 [s]
U_MAX = 10.0   # 구동기가 받을 제어 출력의 최댓값(가정)


def simulate(tf, on_measurement=False, ts=TS, kp=KP, ki=KI, kd=KD, dr=DR):
    """폐루프를 돌려 설정값 r, 측정값 y, 제어 출력 u, 미분 항 d를 돌려준다.

    tf: 미분 필터 시간상수 [s]. 0이면 필터 없음.
    on_measurement=True: 미분 항 입력이 오차 대신 -y(측정값의 변화만 봄).
    """
    n = int(round(T_END / ts))
    k_step = int(round(T_STEP / ts))
    a = np.exp(-ts / TAU)
    alpha = tf / (tf + ts)
    beta = kd / (tf + ts)
    r = np.where(np.arange(n) >= k_step, dr, 0.0)
    y = np.zeros(n)
    u = np.zeros(n)
    d = np.zeros(n)
    integ = x_prev = d_prev = 0.0
    for k in range(n):
        e = r[k] - y[k]
        integ += ki * ts * e
        x = -y[k] if on_measurement else e
        d[k] = alpha * d_prev + beta * (x - x_prev)
        u[k] = kp * e + integ + d[k]
        if k + 1 < n:
            y[k + 1] = a * y[k] + (1.0 - a) * K_PLANT * u[k]
        x_prev, d_prev = x, d[k]
    return r, y, u, d, k_step


def kick_theory(kd, dr, ts, tf):
    """설정값이 바뀐 샘플의 미분 항 Kd*dr/(Tf+Ts)."""
    return kd * dr / (tf + ts)


def first_sample_output(kp, ki, kd, dr, ts, tf):
    """설정값이 바뀐 샘플의 제어 출력 Kp*dr + Ki*Ts*dr + Kd*dr/(Tf+Ts)."""
    return kp * dr + ki * ts * dr + kick_theory(kd, dr, ts, tf)


def tf_min_for_step(u_max, kp, ki, kd, ts, dr):
    """설정값이 바뀐 샘플의 제어 출력을 u_max 이하로 두는 Tf 하한 [s]."""
    return kd * dr / (u_max - kp * dr - ki * ts * dr) - ts


def pulse_area(tf, kd=KD, dr=DR, ts=TS, m=5000):
    """필터에 오차 계단만 넣고 미분 항을 모두 더해 Ts를 곱한 면적(식: Kd*dr)."""
    alpha, beta = tf / (tf + ts), kd / (tf + ts)
    d = x_prev = total = 0.0
    for k in range(m):
        x = dr if k >= 100 else 0.0
        d = alpha * d + beta * (x - x_prev)
        x_prev = x
        total += d * ts
    return total


if __name__ == "__main__":
    tf_lo = tf_min_for_step(U_MAX, KP, KI, KD, TS, DR)
    print(f"제어 출력 한도 {U_MAX:g}에서 구한 Tf 하한 {tf_lo * 1e3:.5f} ms")
    for tf in (0.0, 4e-3, 4.0013e-3):
        _, y, u, d, k0 = simulate(tf)
        print(f"Tf {tf * 1e3:.4f} ms: 미분 항 {d[k0]:.3f} (식 {kick_theory(KD, DR, TS, tf):.3f}), "
              f"제어 출력 최댓값 {u.max():.4f}, 50 ms 뒤 측정값 {y[k0 + 50]:.4f}")
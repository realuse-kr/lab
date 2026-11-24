"""F-5C: 제어 출력이 구동기 한도에서 잘리는 동안 PI의 적분 항이 얼마나 쌓이는지 재 보기.

PI는 병렬형이다: u = Kp*e + I, I[k] = I[k-1] + Ki*Ts*e[k] (지금 샘플의 오차까지 더한다).
e = r - y (r 설정값, y 측정값). 구동기는 u를 -UMAX..UMAX로 잘라 받는다.
제어 대상은 1차 지연 플랜트 tau*y' = -y + K*us이고, 구동기 입력 us는 한 샘플 동안 일정하게 유지한다.
mode: 'ideal'  한도 없음(자르지 않은 u를 그대로 구동기에 넣는 가상의 경우)
      'plain'  한도에서 자르고 적분 항은 그대로 더함
      'stop'   한도에서 자르고, 계산한 u가 한도를 넘으면서 오차가 u와 같은 방향이면 이번 오차를 적분 항에 더하지 않음
"""
import math
import numpy as np

KP = 2.0       # 비례 게인
KI = 6.0       # 적분 게인
TAU = 0.5      # 플랜트 시간상수 [s]
K_PLANT = 1.0  # 플랜트 정상상태 이득
TS = 1e-3      # 샘플링 주기 [s]
DR = 8.0       # 설정값 계단의 크기
T_STEP = 0.1   # 설정값이 바뀌는 시각 [s]
T_END = 4.0    # 시뮬레이션 길이 [s]
UMAX = 10.0    # 구동기가 받을 수 있는 제어 출력의 한도(가정)


def simulate(mode, ki=KI, kp=KP, dr=DR, umax=UMAX, ts=TS):
    """폐루프를 돌려 시간 t, 측정값 y, 계산한 제어 출력 u, 적분 항 integ, 설정값이 바뀐 샘플 번호 k0를 돌려준다."""
    n = int(round(T_END / ts))
    k0 = int(round(T_STEP / ts))
    a = math.exp(-ts / TAU)
    y, u, integ_hist = np.zeros(n), np.zeros(n), np.zeros(n)
    integ = 0.0
    for k in range(n):
        e = (dr if k >= k0 else 0.0) - y[k]
        u_try = kp * e + integ + ki * ts * e
        if mode == "stop" and abs(u_try) > umax and u_try * e > 0:
            pass  # 한도를 넘는 방향이면 이번 오차를 더하지 않는다
        else:
            integ += ki * ts * e
        u[k] = kp * e + integ
        integ_hist[k] = integ
        us = u[k] if mode == "ideal" else min(max(u[k], -umax), umax)
        if k + 1 < n:
            y[k + 1] = a * y[k] + (1.0 - a) * K_PLANT * us
    return np.arange(n) * ts, y, u, integ_hist, k0


def first_sample_output(kp=KP, ki=KI, dr=DR, ts=TS):
    """설정값이 바뀐 샘플의 제어 출력 Kp*dr + Ki*Ts*dr."""
    return kp * dr + ki * ts * dr


def t_reach(dr=DR, umax=UMAX, tau=TAU, k=K_PLANT):
    """제어 출력이 한도에 걸려 있는 동안 측정값이 설정값에 닿는 시각(설정값이 바뀐 때부터) [s]. dr < k*umax일 때만 닿는다."""
    return tau * math.log(k * umax / (k * umax - dr))


def area_a(dr=DR, umax=UMAX, tau=TAU, k=K_PLANT):
    """한도에 걸려 있는 동안 오차를 더한 면적 A = tau*dr - (k*umax - dr)*t_reach."""
    return tau * dr - (k * umax - dr) * t_reach(dr, umax, tau, k)


def windup_formula(ki=KI, dr=DR, umax=UMAX):
    """설정값에 닿는 시점의 적분 항 식 값 Ki*A (연속 시간 식)."""
    return ki * area_a(dr, umax)


def metrics(mode, ki=KI):
    """한 경우의 지표: 오버슈트, 한도를 넘은 시간, 적분 항 최댓값, 설정값에 닿는 시점의 적분 항 등."""
    t, y, u, integ, k0 = simulate(mode, ki=ki)
    cross = k0 + int(np.argmax(y[k0:] >= DR))
    sat_len = int(np.sum(u[k0:] > UMAX))  # 계산한 제어 출력이 한도를 넘은 샘플 수
    band = np.where(np.abs(y[k0:] - DR) > 0.02 * DR)[0]
    return {
        "overshoot": float(y.max() - DR),
        "overshoot_pct": float((y.max() - DR) / DR * 100),
        "cross_ms": float((cross - k0) * TS * 1e3),
        "over_limit_ms": float(sat_len * TS * 1e3),
        "integ_max": float(integ.max()),
        "integ_at_cross": float(integ[cross]),
        "settle_s": float((band[-1] + 1) * TS) if len(band) else 0.0,
    }


if __name__ == "__main__":
    print(f"설정값이 바뀐 샘플의 제어 출력 {first_sample_output():.3f} (한도 {UMAX:g})")
    print(f"한도에 걸려 설정값에 닿는 시각 {t_reach() * 1e3:.1f} ms, A {area_a():.4f}, Ki*A {windup_formula():.3f}")
    for m in ("ideal", "plain", "stop"):
        r = metrics(m)
        print(f"{m}: 오버슈트 {r['overshoot']:.3f}, 한도를 넘은 시간 {r['over_limit_ms']:.0f} ms, "
              f"적분 항 최댓값 {r['integ_max']:.3f}")

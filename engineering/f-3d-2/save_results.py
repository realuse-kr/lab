"""results_f3d2.json을 만든다. 같은 폴더에서 실행한다: python3 save_results.py

저장하는 것
- atol 변화표(rtol=1e-6)와 rtol 변화표(atol=1e-12)
- 수용한 스텝마다의 시각과 시간 스텝 크기(그림 데이터)
- 확인 항목: 함수 평가 횟수 식, 스텝 갱신 규칙을 직접 구현한 코드와의 일치,
  추정 오차가 |x|에 비례하는지, 계수 일치
"""
import json
import numpy as np
import scipy
from scipy.integrate import solve_ivp, RK45
from scipy.integrate._ivp import rk as scipy_rk
from rk45_tolerance import TAU, T, EXACT

# Dormand-Prince 5(4) 계수 (Dormand & Prince 1980)
C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
A = [[],
     [1 / 5],
     [3 / 40, 9 / 40],
     [44 / 45, -56 / 15, 32 / 9],
     [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
     [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
     [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]


def fun(t, x):
    return -x / TAU


def solve_counted(rtol, atol):
    """solve_ivp를 돌리고 우변 함수를 부른 횟수를 직접 센다."""
    calls = []

    def f(t, x):
        calls.append(t)
        return fun(t, x)

    s = solve_ivp(f, (0, T), [1.0], method="RK45", rtol=rtol, atol=atol)
    return s, len(calls)


def own_rk45(rtol, atol, h0):
    """수용 조건과 시간 스텝 갱신 규칙(SciPy RK45와 같은 규칙)을 직접 구현한다.
    반환: 수용한 스텝의 끝 시각, 수용 스텝 수, 거부 스텝 수, 추정 오차/|x|,
    스텝 크기, 추정 오차/허용 오차(en)"""
    t, x, h = 0.0, 1.0, h0
    k1 = fun(t, x)
    ts, accepted, rejected, ratios, hs, ens = [], 0, 0, [], [], []
    while t < T:
        step_rejected = False
        while True:
            t_new = min(t + h, T)
            hh = t_new - t
            k = [k1]
            for i in range(1, 6):
                xi = x + hh * sum(a * kj for a, kj in zip(A[i], k))
                k.append(fun(t + C[i] * hh, xi))
            x_new = x + hh * sum(b * kj for b, kj in zip(B[:6], k))
            k7 = fun(t_new, x_new)
            k.append(k7)
            err = hh * sum(e * kj for e, kj in zip(E, k))
            scale = atol + max(abs(x), abs(x_new)) * rtol
            en = abs(err) / scale
            if en < 1:
                factor = 10 if en == 0 else min(10, 0.9 * en ** (-1 / 5))
                if step_rejected:
                    factor = min(1, factor)
                h = hh * factor
                ts.append(t_new)
                hs.append(hh)
                ratios.append(abs(err) / abs(x))
                ens.append(en)
                accepted += 1
                t, x, k1 = t_new, x_new, k7
                break
            h = hh * max(0.2, 0.9 * en ** (-1 / 5))
            rejected += 1
            step_rejected = True
    return (np.array(ts), accepted, rejected, np.array(ratios), np.array(hs),
            np.array(ens))


def row(rtol, atol):
    s, counted = solve_counted(rtol, atol)
    h = np.diff(s.t)
    attempts = (s.nfev - 2) // 6
    h0 = RK45(fun, 0.0, [1.0], T, rtol=rtol, atol=atol).h_abs
    ts, acc, rej, ratios, hs, ens = own_rk45(rtol, atol, h0)
    return {
        "rtol": rtol, "atol": atol,
        "nfev": int(s.nfev), "nfev_counted_by_wrapper": counted,
        "attempted_steps": attempts, "accepted_steps": len(h),
        "rejected_steps": attempts - len(h),
        "nfev_equals_2_plus_6_attempts": bool(s.nfev == 2 + 6 * attempts),
        "rel_error_at_T": float(abs(s.y[0, -1] - EXACT) / EXACT),
        "initial_h_ms": float(h0 * 1e3),
        "h_first_ms": float(h[0] * 1e3),
        "h_max_ms": float(h.max() * 1e3),
        "h_last_clipped_ms": float(h[-1] * 1e3),
        "h_max_excluding_last_ms": float(h[:-1].max() * 1e3),
        "own_vs_scipy_max_abs_t_diff": float(np.max(np.abs(ts - s.t[1:]))),
        "own_accepted": acc, "own_rejected": rej,
    }, s


results = {"scipy": scipy.__version__, "numpy": np.__version__,
           "TAU_s": TAU, "T_s": T, "exact_x_T": float(EXACT)}

# 1) atol 변화 (rtol = 1e-6)
atol_rows, hist = [], {}
for atol in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2):
    r, s = row(1e-6, atol)
    atol_rows.append(r)
    hist[str(atol)] = {"t_ms": (s.t[1:] * 1e3).tolist(),
                       "h_ms": (np.diff(s.t) * 1e3).tolist()}
results["atol_sweep_rtol_1e-6"] = atol_rows
results["step_history"] = hist

# 2) rtol 변화 (atol = 1e-12)
results["rtol_sweep_atol_1e-12"] = [row(rt, 1e-12)[0]
                                    for rt in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10)]

# 3) 확인 항목
s, _ = solve_counted(1e-6, 1e-12)
h0 = RK45(fun, 0.0, [1.0], T, rtol=1e-6, atol=1e-12).h_abs
_, _, _, ratios, hs, ens = own_rk45(1e-6, 1e-12, h0)
mid = slice(3, -1)  # 처음 세 스텝과 60ms에 맞춰 잘린 마지막 스텝을 뺀다
checks = {
    # SciPy의 E는 부호만 반대(4차 해 - 5차 해)이고 오차의 절댓값에는 영향이 없다
    "coefficients_E_equal_scipy_up_to_sign": bool(
        np.allclose(E, -scipy_rk.RK45.E, rtol=0, atol=1e-15)),
    "coefficients_B_equal_scipy": bool(
        np.allclose(B[:6], scipy_rk.RK45.B, rtol=0, atol=1e-15)),
    "plateau_steps": int(len(hs[mid])),
    "plateau_h_ms_min": float(hs[mid].min() * 1e3),
    "plateau_h_ms_max": float(hs[mid].max() * 1e3),
    "plateau_h_over_tau_min": float(hs[mid].min() / TAU),
    "plateau_h_over_tau_max": float(hs[mid].max() / TAU),
    "plateau_err_over_x_min": float(ratios[mid].min()),
    "plateau_err_over_x_max": float(ratios[mid].max()),
    "plateau_en_min": float(ens[mid].min()),
    "plateau_en_max": float(ens[mid].max()),
    "fixed_point_en_0.9^5": 0.9 ** 5,
    "solve_ivp_t_are_accepted_steps_only": all(
        r["accepted_steps"] == len(h["t_ms"]) for r, h in
        zip(atol_rows, hist.values())),
}

def start_h(key):
    """수용한 스텝의 시작 시각(ms)과 크기(ms). 60ms에 맞춰 잘린 마지막 스텝은 뺀다."""
    t = np.array(hist[key]["t_ms"])
    h = np.array(hist[key]["h_ms"])
    return np.concatenate([[0.0], t[:-1]])[:-1], h[:-1]


# 같은 시각에서 atol이 큰 쪽의 스텝 크기가 항상 큰지(겹치는 시각 범위, 그림의 네 값)
keys = [str(a) for a in (1e-12, 1e-8, 1e-6, 1e-4)]
grid = np.linspace(0, 50, 501)
order_ok, order_min_gap = True, []
for lo, hi in zip(keys[:-1], keys[1:]):
    (t_lo, h_lo), (t_hi, h_hi) = start_h(lo), start_h(hi)
    m = grid <= min(t_lo.max(), t_hi.max())
    gap = np.interp(grid, t_hi, h_hi)[m] - np.interp(grid, t_lo, h_lo)[m]
    order_ok = order_ok and bool(np.all(gap > 0))
    order_min_gap.append(float(gap.min()))
checks["bigger_atol_gives_bigger_h_at_same_time_1e-12_to_1e-4"] = order_ok
checks["min_h_gap_ms_between_neighbor_atol"] = order_min_gap
results["checks"] = checks

with open("results_f3d2.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, ensure_ascii=False, indent=1)
print("saved results_f3d2.json")

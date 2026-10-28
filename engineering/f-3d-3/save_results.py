"""results_f3d3.json을 만든다. 본문 표와 그림의 숫자, 확인 항목의 결과를 담는다."""
import json, os, sys
import numpy as np, scipy
from scipy.integrate import solve_ivp

TAU, T = 0.010, 0.060
EXACT = np.exp(-T / TAU)
f = lambda t, x: -x / TAU
KW = dict(method="RK45", rtol=1e-6, atol=1e-12)
DTS = (0.010, 0.005, 0.002, 0.001, 0.0005, 0.0001)
rel = lambda x, t: np.abs(x - np.exp(-t / TAU)) / np.exp(-t / TAU)


def rk4_grid(h):
    n = round(T / h)
    x, xs = 1.0, [1.0]
    for _ in range(n):
        k1 = f(0, x); k2 = f(0, x + h / 2 * k1)
        k3 = f(0, x + h / 2 * k2); k4 = f(0, x + h * k3)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return 4 * n, np.array(xs), np.linspace(0, T, n + 1)


base = solve_ivp(f, (0, T), [1.0], dense_output=True, **KW)
hb = np.diff(base.t)
R = {
    "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
    "model": "dx/dt = -x/tau, tau = 0.010 s, x(0) = 1, t in [0, 0.060] s, rtol 1e-6, atol 1e-12",
    "exact_at_T": EXACT,
    "baseline_no_t_eval_no_max_step": {
        "nfev": base.nfev, "accepted_steps": len(hb), "attempts": (base.nfev - 2) // 6,
        "rejected": (base.nfev - 2) // 6 - len(hb), "output_points": len(base.t),
        "h_ms": (hb * 1e3).tolist(), "h_max_excluding_last_ms": hb[:-1].max() * 1e3,
        "h_min_ms": hb.min() * 1e3, "h_last_ms": hb[-1] * 1e3,
    },
    "rows": [],
}
chk = dict(t_eval_nfev_equals_baseline=True, t_eval_step_times_equal_baseline=True,
           t_eval_values_equal_dense_output=True, t_eval_max_error_at_T=True,
           max_step_large_same_as_baseline=True, max_step_small_all_steps_equal_max_step=True,
           max_step_nfev_is_2_plus_6_steps_when_no_rejection=True, rk4_nfev_is_4n=True)

for dt in DTS:
    n = round(T / dt)
    te = np.linspace(0, T, n + 1)
    a = solve_ivp(f, (0, T), [1.0], t_eval=te, dense_output=True, **KW)
    ra = rel(a.y[0], te)
    chk["t_eval_nfev_equals_baseline"] &= a.nfev == base.nfev
    chk["t_eval_step_times_equal_baseline"] &= bool(np.array_equal(a.sol.ts, base.sol.ts))
    chk["t_eval_values_equal_dense_output"] &= float(np.abs(a.y[0] - a.sol(te)[0]).max()) == 0.0
    chk["t_eval_max_error_at_T"] &= int(ra.argmax()) == n
    b = solve_ivp(f, (0, T), [1.0], max_step=dt, **KW)
    hm = np.diff(b.t)
    attempts = (b.nfev - 2) / 6
    if dt >= 0.005:
        chk["max_step_large_same_as_baseline"] &= bool(np.array_equal(b.t, base.t)) and b.nfev == base.nfev
    else:
        chk["max_step_small_all_steps_equal_max_step"] &= bool(np.allclose(hm, dt, rtol=0, atol=1e-12))
        chk["max_step_nfev_is_2_plus_6_steps_when_no_rejection"] &= b.nfev == 2 + 6 * len(hm)
    nf4, x4, t4 = rk4_grid(dt)
    r4 = rel(x4, t4)
    chk["rk4_nfev_is_4n"] &= nf4 == 4 * n
    R["rows"].append({
        "dt_out_ms": dt * 1e3, "n_output_points": n + 1,
        "t_eval": {"nfev": a.nfev, "rel_error_at_T": ra[-1], "max_rel_error_output_points": ra.max(),
                   "max_rel_error_interior_points": ra[:-1].max()},
        "max_step": {"nfev": b.nfev, "accepted_steps": len(hm), "attempts": attempts,
                     "output_points": len(b.t), "rel_error_at_T": abs(b.y[0, -1] - EXACT) / EXACT,
                     "output_on_requested_grid": bool(np.allclose(b.t / dt, np.round(b.t / dt), atol=1e-6))},
        "rk4_fixed_time_step": {"nfev": nf4, "rel_error_at_T": r4[-1], "max_rel_error_grid": r4.max()},
    })

R["checks"] = chk
R["figure_data_ms"] = {  # make_figures_f3d3.py가 읽는다
    "rk45_step_boundaries": (base.t * 1e3).tolist(),
    "max_step_1ms_boundaries": (solve_ivp(f, (0, T), [1.0], max_step=0.001, **KW).t * 1e3).tolist(),
    "grid_1ms": (np.linspace(0, T, 61) * 1e3).tolist(),
    "grid_10ms": (np.linspace(0, T, 7) * 1e3).tolist(),
}

# 연습문제 정답
ex = {}
for ms in (np.inf, 2.4e-3, 2.3e-3, 2.0e-3):
    s = solve_ivp(f, (0, T), [1.0], max_step=ms, **KW)
    ex[f"max_step_{ms*1e3:g}ms"] = {"nfev": s.nfev, "accepted_steps": len(s.t) - 1,
                                     "attempts": (s.nfev - 2) // 6}
s = solve_ivp(f, (0, T), [1.0], method="RK45", rtol=1e-3, atol=1e-12, t_eval=np.linspace(0, T, 601))
ex["rtol_1e-3_t_eval_0.1ms"] = {"nfev": s.nfev, "rel_error_at_T": abs(s.y[0, -1] - EXACT) / EXACT}
s = solve_ivp(f, (0, T), [1.0], method="RK45", rtol=1e-3, atol=1e-12, max_step=1e-4)
ex["rtol_1e-3_max_step_0.1ms"] = {"nfev": s.nfev}
nf, x, t = rk4_grid(0.00025)
ex["rk4_h_0.25ms"] = {"nfev": nf, "rel_error_at_T": abs(x[-1] - EXACT) / EXACT}
R["exercises"] = ex

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_f3d3.json")
json.dump(R, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("checks:", chk)
assert all(chk.values()), "확인 항목 실패"

import json
import euler_vs_rk4 as m

T, TAU_A, A_60 = m.T, m.TAU_A, m.A_60
STEPS = {"euler": m.euler, "rk4": m.rk4}
CALLS_PER_STEP = {"euler": 1, "rk4": 4}


def min_n(pred, step, tau_b):
    lo = hi = 1
    while not pred(step, hi, tau_b):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if pred(step, mid, tau_b) else (mid + 1, hi)
    return lo


def acc_ok(step, n, tau_b):
    return abs(m.solve(step, n, TAU_A) - A_60) / A_60 <= 0.01


def stab_ok(step, n, tau_b):
    return abs(m.solve(step, n, tau_b)) < 1


def calls(name, n):
    return 2 * n * CALLS_PER_STEP[name]


def row(tau_b_ms):
    tau_b = tau_b_ms / 1000
    out = {"tau_b_ms": tau_b_ms}
    for name, step in STEPS.items():
        n_acc = min_n(acc_ok, step, tau_b)
        n_stab = min_n(stab_ok, step, tau_b)
        n = m.min_steps(step, tau_b)
        out[name] = {
            "steps_accuracy_only": n_acc, "steps_stability_only": n_stab,
            "steps": n, "h_ms": 1000 * T / n, "calls": calls(name, n),
            "binding": "accuracy" if n_acc >= n_stab else "stability",
            "a_rel_error": abs(m.solve(step, n, TAU_A) - A_60) / A_60,
        }
    out["ratio_rk4_over_euler"] = out["rk4"]["calls"] / out["euler"]["calls"]
    return out


# RK4 stability limit: |growth factor| = 1 on the negative real axis
lo, hi = 2.0, 3.0
for _ in range(100):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if abs(m.rk4(1.0, mid, 1.0)) < 1 else (lo, mid)
z_limit = lo

# growth factor of rk4 equals 1 - z + z^2/2 - z^3/6 + z^4/24
poly_diff = max(abs(m.rk4(1.0, z, 1.0) - (1 - z + z**2 / 2 - z**3 / 6 + z**4 / 24))
                for z in [i / 100 for i in range(1, 400)])

# crossover: largest tau_B (ms) where RK4 needs more calls than Euler
a, b = 0.02, 0.2
for _ in range(40):
    c = (a + b) / 2
    more = m.min_steps(m.rk4, c / 1000) * 8 > m.min_steps(m.euler, c / 1000) * 2
    a, b = (c, b) if more else (a, c)

def crossover_ms(tol):
    """Largest tau_B (ms) at which RK4 needs more calls than Euler, for a given A tolerance."""
    def ok_tol(step, n, tau_b):
        return acc_tol(step, n, tol) and stab_ok(step, n, tau_b)
    def total(name, tau_b):
        return calls(name, min_n(ok_tol, STEPS[name], tau_b))
    a, b = 0.001, 5.0
    for _ in range(25):
        c = (a * b) ** 0.5
        more = total("rk4", c / 1000) > total("euler", c / 1000)
        a, b = (c, b) if more else (a, c)
    return (a * b) ** 0.5


def acc_tol(step, n, tol):
    return abs(m.solve(step, n, TAU_A) - A_60) / A_60 <= tol


# just below and above the RK4 limit for equation B (h = 5.5ms and 5.6ms, 15 steps)
edge = []
for h_ms in (5.5, 5.6):
    x = 1.0
    for _ in range(15):
        x = m.rk4(x, h_ms / 1000, 0.002)
    edge.append({"h_ms": h_ms, "steps": 15, "b_after_15": abs(x)})

out = {
    "model": "dx/dt = -x/tau, two independent equations, x(0) = 1, virtual math model",
    "tau_a_ms": 10, "time_ms": 60, "target_rel_error_a_60ms": 0.01,
    "rk4_stability_limit_z": z_limit,
    "rk4_stability_limit_b2ms_ms": z_limit * 2,
    "euler_stability_limit_b2ms_ms": 4,
    "rk4_growth_factor_polynomial_max_diff": poly_diff,
    "asymptotic_ratio_when_both_stability_limited": 8 / z_limit,
    "crossover_tau_b_ms": (a + b) / 2,
    "crossover_tau_b_ms_by_tolerance": {str(t): crossover_ms(t) for t in (0.1, 0.01, 0.001)},
    "steps_accuracy_only_by_tolerance": {
        str(t): {name: min_n(lambda s, n, tb, t=t: acc_tol(s, n, t), STEPS[name], 0.002)
                 for name in STEPS}
        for t in (0.1, 0.01, 0.001)},
    "rows": [row(t) for t in (2, 0.2, 0.02, 0.002)],
    "rk4_edge_b2ms": edge,
}
with open("results_f3d.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

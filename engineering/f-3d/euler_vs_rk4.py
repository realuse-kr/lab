T, TAU_A, A_60 = 0.060, 0.010, 2.718281828459045 ** -6

def euler(x, h, tau):
    return x + h * (-x / tau)
def rk4(x, h, tau):
    k1 = -x / tau
    k2 = -(x + h / 2 * k1) / tau
    k3 = -(x + h / 2 * k2) / tau
    k4 = -(x + h * k3) / tau
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def solve(step, n, tau):
    x = 1.0
    for _ in range(n):
        x = step(x, T / n, tau)
    return x
def ok(step, n, tau_b):
    err = abs(solve(step, n, TAU_A) - A_60) / A_60
    return err <= 0.01 and abs(solve(step, n, tau_b)) < 1
def min_steps(step, tau_b):
    lo = hi = 1
    while not ok(step, hi, tau_b):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(step, mid, tau_b) else (mid + 1, hi)
    return lo

for ms in (2, 0.2, 0.02, 0.002):
    n_e, n_r = (min_steps(s, ms / 1000) for s in (euler, rk4))
    print(f"tau_B={ms}ms  Euler {2 * n_e} calls  RK4 {8 * n_r} calls")

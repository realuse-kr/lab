import json, math

TAU_SLOW = 0.010
TAU_FAST = 0.002
T = 0.060


def run(dt, tau_list, steps):
    vals = [1.0 for _ in tau_list]
    for _ in range(steps):
        vals = [v + dt * (-v / tau) for v, tau in zip(vals, tau_list)]
    return vals


rows = []
for dt_ms in [1, 3, 4, 5, 10, 15]:
    dt = dt_ms / 1000
    slow, fast = run(dt, [TAU_SLOW, TAU_FAST], round(T / dt))
    rows.append({"dt_ms": dt_ms, "m_slow": 1 - dt / TAU_SLOW, "m_fast": 1 - dt / TAU_FAST,
                 "abs_fast": abs(fast), "abs_slow": abs(slow)})

edge = []
for dt_ms in [3.9, 4.0, 4.1]:
    dt = dt_ms / 1000
    fast, = run(dt, [TAU_FAST], 15)
    edge.append({"dt_ms": dt_ms, "steps": 15, "abs_fast": abs(fast)})

out = {
    "tau_slow_ms": 10, "tau_fast_ms": 2, "time_ms": 60,
    "limit_slow_ms": 2 * 10, "limit_fast_ms": 2 * 2,
    "step_count_ratio": (2 * 10) / (2 * 2),
    "true_slow_at_60ms": math.exp(-T / TAU_SLOW),
    "true_fast_at_60ms": math.exp(-T / TAU_FAST),
    "true_fast_at_8ms": math.exp(-0.008 / TAU_FAST),
    "rows": rows, "edge": edge,
}
with open("results_f3c2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:300])

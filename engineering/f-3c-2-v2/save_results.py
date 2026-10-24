import json
import math

TAU_A = 0.010
TAU_B = 0.002
T = 0.060


def euler(h, tau, steps):
    x = 1.0
    for _ in range(steps):
        x = x + h * (-x / tau)
    return x


rows = []
for h_ms in [1, 3, 4, 5, 10, 15]:
    h = h_ms / 1000
    n = round(T / h)
    rows.append({
        "h_ms": h_ms, "steps": n,
        "g_a": 1 - h / TAU_A, "g_b": 1 - h / TAU_B,
        "a_60ms": euler(h, TAU_A, n), "b_60ms": euler(h, TAU_B, n),
    })

edge = []
for h_ms in [3.9, 4.0, 4.1]:
    h = h_ms / 1000
    b = euler(h, TAU_B, 15)
    edge.append({"h_ms": h_ms, "steps": 15, "g_b": 1 - h / TAU_B, "b_after_15": b})

out = {
    "model": "dx/dt = -x/tau, two independent equations, x(0) = 1, virtual math model",
    "tau_a_ms": 10, "tau_b_ms": 2, "time_ms": 60,
    "stability_limit_a_ms": 2 * 10, "stability_limit_b_ms": 2 * 2,
    "limit_ratio": (2 * 10) / (2 * 2),
    "exact_a_60ms": math.exp(-T / TAU_A),
    "exact_b_60ms": math.exp(-T / TAU_B),
    "exact_b_8ms": math.exp(-0.008 / TAU_B),
    "rows": rows, "edge": edge,
}
with open("results_f3c2v2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

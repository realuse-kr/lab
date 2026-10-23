"""growth_vs_dt.py와 같은 계산을 results_f3c.json에 저장한다.

표의 여섯 간격, 그림에 쓰는 15·20·25ms의 걸음별 크기, 한계 간격(2 x TAU),
연습문제 정답(TAU = 0.050)을 함께 넣는다.
"""
import json

STEPS = 10


def run(tau, dt, steps):
    y, ys = 1.0, [1.0]
    for _ in range(steps):
        y = y + dt * (-y / tau)
        ys.append(y)
    return ys


tau = 0.010
rows = []
for dt_ms in [5, 10, 19, 20, 21, 30]:
    dt = dt_ms / 1000
    rows.append({"dt_ms": dt_ms, "dt_over_tau": round(dt / tau, 2),
                 "m": round(1 - dt / tau, 2),
                 "abs_y_after_10": round(abs(run(tau, dt, STEPS)[-1]), 3)})

curve = {}
for dt_ms in [15, 20, 25]:
    ys = run(tau, dt_ms / 1000, 20)
    curve[str(dt_ms)] = [abs(v) for v in ys]

ex = {}
tau2 = 0.050
for dt_ms in [99, 101]:
    ex[str(dt_ms)] = round(abs(run(tau2, dt_ms / 1000, STEPS)[-1]), 3)

out = {"tau_ms": 10, "steps_table": STEPS, "limit_ms": 2 * 1000 * tau,
       "rows": rows, "curve_abs_y": curve,
       "growth_20_steps_dt25": round(curve["25"][-1], 1),
       "exercise_tau50ms": ex}
with open("results_f3c.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:600])

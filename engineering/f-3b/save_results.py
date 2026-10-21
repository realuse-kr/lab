"""같은 계산(overshoot_vs_dt.py와 같은 값)을 json으로 저장하고 오차가 TOL이 되는 간격을 찾는다."""
import json
import numpy as np

WN, ZETA = 10.0, 0.3
T_END = 2.0
TOL = 1.0
EXACT = 100 * np.exp(-np.pi * ZETA / np.sqrt(1 - ZETA**2))
DTS = [0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.02, 0.04]


def error_pp(dt):
    y, v, peak = 0.0, 0.0, 0.0
    for _ in range(int(round(T_END / dt))):
        a = WN**2 * (1.0 - y) - 2 * ZETA * WN * v
        y, v = y + dt * v, v + dt * a
        peak = max(peak, y)
    return 100 * (peak - 1.0) - EXACT


lo, hi = 0.0005, 0.005
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if error_pp(mid) <= TOL else (lo, mid)

rows = [{"dt_ms": dt * 1000, "overshoot_pct": round(EXACT + error_pp(dt), 2),
         "error_pp": round(error_pp(dt), 2), "within_tol": bool(abs(error_pp(dt)) <= TOL)}
        for dt in DTS]
out = {"wn": WN, "zeta": ZETA, "exact_overshoot_pct": round(EXACT, 2), "tol_pp": TOL,
       "dt_limit_ms": round(lo * 1000, 3), "rows": rows,
       "halving_check": {"error_at_1ms": round(error_pp(0.001), 3),
                         "error_at_0p5ms": round(error_pp(0.0005), 3)}}
with open("results_f3b.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))

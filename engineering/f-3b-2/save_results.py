"""overshoot_ts_vs_dt.py와 같은 계산을 json으로 저장하고, 간격을 TS/1000까지 줄인 값과 비교한다.

차이(%p)는 표에 보이는 소수 둘째 자리 값끼리 뺀다.
"""
import json
import numpy as np

WN, ZETA, KP, T_END = 10.0, 0.3, 3.0, 2.0
ZETA_CL = ZETA / np.sqrt(1 + KP)
EXACT = 100 * np.exp(-np.pi * ZETA_CL / np.sqrt(1 - ZETA_CL**2))
Y_END = KP / (1 + KP)
TS_LIST = [0.001, 0.002, 0.005, 0.01]


def overshoot(ts, dt):
    y, v, peak = 0.0, 0.0, 0.0
    for _ in range(int(round(T_END / ts))):
        u = KP * (1.0 - y)
        for _ in range(int(round(ts / dt))):
            a = WN**2 * (u - y) - 2 * ZETA * WN * v
            y, v = y + dt * v, v + dt * a
            peak = max(peak, y)
    return 100 * (peak / Y_END - 1.0)


rows = []
exact_r = round(EXACT, 2)
for ts in TS_LIST:
    same = round(overshoot(ts, ts), 2)
    fine = round(overshoot(ts, ts / 100), 2)
    finest = round(overshoot(ts, ts / 1000), 2)
    rows.append({
        "ts_ms": ts * 1000,
        "dt_equals_ts_pct": same,
        "dt_ts_over_100_pct": fine,
        "dt_ts_over_1000_pct": finest,
        "period_effect_pp": round(fine - exact_r, 2),
        "extra_numeric_error_pp": round(same - fine, 2),
        "gap_100_vs_1000_pp": round(abs(fine - finest), 2),
    })
out = {"wn": WN, "zeta": ZETA, "kp": KP, "exact_continuous_overshoot_pct": exact_r, "rows": rows}
with open("results_f3b2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))

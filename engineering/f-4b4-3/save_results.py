"""F-4B4.3 결과를 results_f4b4_3.json으로 저장하고 닫힌 식과 대조한다."""
import json
import numpy as np
from phase_margin_compute_time import (crossover_and_margin, loop_base, margin_compute,
                                       margin_total, margin_disc, W_HI)

wc0, pm0 = crossover_and_margin(loop_base, W_HI)
res = {"wc0": wc0, "pm0": pm0, "T20": [], "T10_budget": {}, "grid": []}
T = 0.020
for tc_ms in (0, 5, 10, 15, 20):
    tc = tc_ms / 1000
    res["T20"].append({"tc_ms": tc_ms,
        "pm_compute": margin_compute(T, tc)[1], "pm_total": margin_total(T, tc)[1],
        "pm_disc": margin_disc(T, tc)[1], "pred_deg": float(np.degrees(wc0 * (T / 2 + tc)))})
for T_ms in (5, 10, 20):
    for f in (0, 0.5, 1.0):
        T = T_ms / 1000; tc = f * T
        res["grid"].append({"T_ms": T_ms, "tc_ms": T_ms * f,
            "drop_compute": pm0 - margin_compute(T, tc)[1], "drop_disc": pm0 - margin_disc(T, tc)[1],
            "pred_deg": float(np.degrees(wc0 * (T / 2 + tc)))})
# T = 10 ms에서 위상 여유 목표를 지키는 최대 tc (0.1 ms 단위)
for goal in (60, 50, 45):
    T = 0.010
    ok = [k for k in range(0, 101) if margin_disc(T, k / 10000)[1] >= goal]
    res["T10_budget"][str(goal)] = max(ok) / 10 if ok else None
# 범위 점검: T 1~20 ms, tc = 0, T/4, T/2, 3T/4, T 모두에서 어긋남의 최댓값
mx_c = mx_d = 0.0
for t in range(1, 21):
    T = t / 1000
    for f in (0, .25, .5, .75, 1.0):
        tc = f * T; pred = np.degrees(wc0 * (T / 2 + tc))
        mx_c = max(mx_c, abs(pm0 - margin_compute(T, tc)[1] - pred))
        mx_d = max(mx_d, abs(pm0 - margin_disc(T, tc)[1] - pred))
res["max_dev_compute_deg"] = float(mx_c); res["max_dev_disc_deg"] = float(mx_d)
res["check_wc0"] = bool(abs(wc0 - 24.02) < 0.01)
res["check_pm0"] = bool(abs(pm0 - 72.8) < 0.05)
res["check_compute_vs_total"] = bool(mx_c < 0.3)
res["check_disc_vs_pred"] = bool(mx_d < 1.3)
with open("results_f4b4_3.json", "w") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1)
print({k: v for k, v in res.items() if k.startswith(("check", "max", "T10"))})

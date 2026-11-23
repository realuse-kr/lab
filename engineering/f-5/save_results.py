"""F-5 결과를 results_f5.json으로 저장하고 scipy.signal.step, 극점, 변환 왕복과 대조한다."""
import json
import numpy as np
from scipy import signal
import pid_form_mismatch as m

CASES = {
    "right": m.ideal_to_parallel(2.0, 0.1, 0.02),    # 이상형 숫자를 이상형으로
    "ideal_into_parallel": (2.0, 0.1, 0.02),         # 이상형 숫자를 병렬형에 그대로
    "parallel_into_ideal": m.ideal_to_parallel(2.0, 20.0, 0.04),  # 병렬형 숫자를 이상형에 그대로
    "kd_only": (2.0, 20.0, 0.02),                    # Ki는 맞고 Kd만 Td 그대로
}
out = {"plant": "1/((0.1 s+1)(0.02 s+1))", "target": 1.0, "cases": {}}
for name, (kp, ki, kd) in CASES.items():
    long = name in ("ideal_into_parallel", "parallel_into_ideal")
    t = np.linspace(0, 600.0 if long else 1.0, 2_000_001 if long else 100001)
    y = m.step_response(kp, ki, kd, t)
    os_, ts = m.metrics(t, y)
    num, den = m.closed_loop(kp, ki, kd)
    t6 = np.linspace(0, 0.6, 60001)
    y6 = m.step_response(kp, ki, kd, t6)
    y_sci = signal.step(signal.TransferFunction(num, den), T=t6)[1]
    poles = np.roots(den)
    out["cases"][name] = {
        "Kp": kp, "Ki": ki, "Kd": kd,
        "y_at_0.6s": round(float(y6[-1]), 4),
        "overshoot_pct": round(float(os_), 2),
        "settling_2pct_s": round(float(ts), 4),
        "max_diff_vs_scipy_step": float(np.abs(y6 - y_sci).max()),
        "stable": bool(np.all(poles.real < 0)),
        "slowest_pole": round(float(poles.real.max()), 4),
    }
out["p_only_final_value"] = round(2.0 / (1.0 + 2.0), 4)
rt = m.parallel_to_ideal(*m.ideal_to_parallel(2.0, 0.1, 0.02))
out["round_trip_ideal"] = [round(float(v), 12) for v in rt]
out["ki_ratio_right_vs_wrong"] = 20.0 / 0.1
with open("results_f5.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

"""F-5B.3 결과를 results_f5b3.json으로 저장한다."""
import json
import numpy as np
from pid_derivative_filter import (KP, KI, KD, TS, DR, U_MAX, simulate, kick_theory,
                                   first_sample_output, tf_min_for_step, pulse_area)

res = {"params": {"Kp": KP, "Ki": KI, "Kd": KD, "Ts_ms": TS * 1e3, "dr": DR, "u_max": U_MAX}}
tf_lo = tf_min_for_step(U_MAX, KP, KI, KD, TS, DR)
res["tf_min_ms"] = tf_lo * 1e3
res["N_at_tf_min"] = (KD / KP) / tf_lo

rows = []
for tf_ms in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
    tf = tf_ms * 1e-3
    _, y, u, d, k0 = simulate(tf)
    rows.append({"tf_ms": tf_ms, "d_at_step": float(d[k0]), "d_theory": kick_theory(KD, DR, TS, tf),
                 "u_max": float(u.max()), "u_first_theory": first_sample_output(KP, KI, KD, DR, TS, tf),
                 "u_argmax_is_step_sample": bool(u.argmax() == k0), "y_50ms": float(y[k0 + 50]),
                 "d_next": float(d[k0 + 1]), "u_min_after": float(u[k0:].min()),
                 "area_filter_only": pulse_area(tf)})
res["rows"] = rows

tf_ok = 4.0013e-3  # 하한 4.00125 ms를 올린 값
_, y, u, d, k0 = simulate(tf_ok)
res["at_tf_ok"] = {"tf_ms": tf_ok * 1e3, "d_at_step": float(d[k0]), "u_max": float(u.max()),
                   "y_50ms": float(y[k0 + 50]), "d_next": float(d[k0 + 1]), "u_min_after": float(u[k0:].min())}
_, y, u, d, k0 = simulate(0.0, kd=0.008)
res["kd_008_no_filter"] = {"d_at_step": float(d[k0]), "u_max": float(u.max()), "y_50ms": float(y[k0 + 50])}
_, y, u, d, k0 = simulate(0.0, on_measurement=True)
res["measurement_derivative"] = {"d_at_step": float(d[k0]), "u_max": float(u.max()), "y_50ms": float(y[k0 + 50])}

ts_rows = []
for ts_ms in (4.0, 2.0, 1.0, 0.5):
    ts = ts_ms * 1e-3
    _, _, _, d0, k0 = simulate(0.0, ts=ts)
    _, _, _, d4, k4 = simulate(4e-3, ts=ts)
    ts_rows.append({"ts_ms": ts_ms, "no_filter": float(d0[k0]), "tf4_code": float(d4[k4]),
                    "tf4_theory": kick_theory(KD, DR, ts, 4e-3)})
res["ts_rows"] = ts_rows

with open("results_f5b3.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
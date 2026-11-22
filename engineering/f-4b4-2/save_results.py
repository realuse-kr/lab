"""F-4B4.2: phase_margin_zoh.py의 결과를 results_f4b4_2.json으로 저장하고, 다른 경로로 다시 구해 대조한다."""
import json
import numpy as np
from phase_margin_zoh import (KP, KI, TAU, W_HI, crossover_and_margin, loop_base,
                              margin_zoh, margin_delay, margin_disc, zoh)

GRID_MS = list(range(1, 41))  # 샘플링 주기 T를 1 ms부터 40 ms까지 1 ms 단위로 계산한다

wc0, pm0 = crossover_and_margin(loop_base, W_HI)

rows = {}
for t_ms in GRID_MS:
    T = t_ms / 1000
    w_z, pm_z = margin_zoh(T)
    w_h, pm_h = margin_delay(T / 2)
    w_d, pm_d = margin_disc(T)
    rows[str(t_ms)] = {
        "wc_zoh": w_z, "pm_zoh_deg": pm_z, "drop_zoh_deg": pm0 - pm_z,
        "wc_zoh_times_half_T_deg": float(np.degrees(w_z * T / 2)),
        "wc0_times_half_T_deg": float(np.degrees(wc0 * T / 2)),
        "pm_half_delay_deg": pm_h, "wc_half_delay": w_h,
        "pm_disc_deg": pm_d, "drop_disc_deg": pm0 - pm_d, "wc_disc": w_d,
    }


def max_abs(f):
    return float(max(abs(f(v)) for v in rows.values()))


# ZOH의 위상과 크기: 각주파수 0 < w < 2 pi/T에서 위상은 -wT/2, 크기는 |sin(wT/2)/(wT/2)|이다.
zoh_phase_err, zoh_mag_err = 0.0, 0.0
for t_ms in GRID_MS:
    T = t_ms / 1000
    w = np.linspace(1e-3, 0.999 * 2 * np.pi / T, 20001)
    h = zoh(w, T)
    zoh_phase_err = max(zoh_phase_err, float(np.max(np.abs(np.angle(h) + w * T / 2))))
    x = w * T / 2
    zoh_mag_err = max(zoh_mag_err, float(np.max(np.abs(np.abs(h) - np.abs(np.sin(x) / x)))))

out = {
    "plant_tau_s": TAU, "kp": KP, "ki": KI,
    "continuous_no_zoh": {"wc_rad_per_s": wc0, "pm_deg": pm0},
    "grid_ms": GRID_MS,
    "rows": rows,
    "max_pm_diff_zoh_vs_half_delay_deg": max_abs(lambda v: v["pm_zoh_deg"] - v["pm_half_delay_deg"]),
    "max_pm_diff_disc_vs_zoh_deg": max_abs(lambda v: v["pm_disc_deg"] - v["pm_zoh_deg"]),
    "max_drop_diff_disc_vs_wc0_half_T_deg": max_abs(lambda v: v["drop_disc_deg"] - v["wc0_times_half_T_deg"]),
    "max_drop_diff_zoh_vs_wc0_half_T_deg": max_abs(lambda v: v["drop_zoh_deg"] - v["wc0_times_half_T_deg"]),
    "zoh_phase_max_err_rad": zoh_phase_err, "zoh_mag_max_err": zoh_mag_err,
}

# 점검 1: ZOH의 위상은 -wT/2이고 크기는 sinc다(각주파수 0 < w < 2 pi/T, T 1~40 ms).
out["check_zoh_phase_is_minus_wT_half"] = bool(zoh_phase_err < 1e-9)
out["check_zoh_magnitude_is_sinc"] = bool(zoh_mag_err < 1e-9)

# 점검 2: ZOH를 넣은 연속 시간 루프의 위상 여유는 "지연 없는 개루프를 새 교차 각주파수에서 본 위상 여유 - 새 교차 각주파수 x T/2"와 같다.
#         지연 없는 개루프의 위상 여유 식은 90도 + atan(KP w/KI) - atan(TAU w)다.
def pm_base_formula(w):
    return float(90.0 + np.degrees(np.arctan(KP * w / KI)) - np.degrees(np.arctan(TAU * w)))


out["check_zoh_pm_equals_base_pm_minus_wc_half_T"] = bool(all(
    abs(v["pm_zoh_deg"] - (pm_base_formula(v["wc_zoh"]) - v["wc_zoh_times_half_T_deg"])) < 1e-5 for v in rows.values()))

# 점검 2b: 줄어든 각도와 지연 없는 교차 각주파수 x T/2의 차이가 1도 이내다.
out["check_zoh_drop_close_to_wc0_half_T_1deg"] = bool(out["max_drop_diff_zoh_vs_wc0_half_T_deg"] < 1.0)

# 점검 3: ZOH는 크기를 1보다 작게 만들어서 교차 각주파수가 지연 없는 값 24.02 rad/s보다 낮아진다.
out["check_zoh_lowers_crossover"] = bool(all(v["wc_zoh"] < wc0 for v in rows.values()))

# 점검 4: ZOH를 넣은 루프와 길이 T/2인 지연을 넣은 루프의 위상 여유 차이가 1도 이내다.
out["check_zoh_close_to_half_delay_1deg"] = bool(out["max_pm_diff_zoh_vs_half_delay_deg"] < 1.0)

# 점검 5: 이산 루프(Tustin + ZOH 플랜트, 지연 없음)와 ZOH를 넣은 연속 시간 루프의 위상 여유 차이가 1도 이내다.
out["check_disc_close_to_zoh_loop_1deg"] = bool(out["max_pm_diff_disc_vs_zoh_deg"] < 1.0)

# 점검 6: 교차 각주파수를 |L(jw)| = 1의 닫힌 식으로 다시 구한다(F-4B4와 같은 값).
w2 = np.roots([TAU ** 2, 1 - KP ** 2, -KI ** 2])
wc_closed = float(np.sqrt(max(w2[np.isreal(w2)].real)))
out["check_crossover_matches_closed_form"] = bool(abs(wc_closed - wc0) < 1e-9)

# 위상 여유 목표별로 허용되는 샘플링 주기: ZOH를 넣은 연속 시간 루프에서 위상 여유가 목표 이상인 가장 큰 T
for goal in (60.0, 50.0):
    ok = [t for t in GRID_MS if rows[str(t)]["pm_zoh_deg"] >= goal]
    out[f"largest_T_ms_pm_zoh_ge_{int(goal)}"] = max(ok) if ok else None

with open("results_f4b4_2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k.startswith("check_")})
print("wc0", round(wc0, 4), "pm0", round(pm0, 4))
print("최대 차이 [도]: ZOH 대 T/2 지연", round(out["max_pm_diff_zoh_vs_half_delay_deg"], 3),
      "| 이산 대 ZOH 루프", round(out["max_pm_diff_disc_vs_zoh_deg"], 3),
      "| ZOH 줄어든 각도 대 wc0*T/2", round(out["max_drop_diff_zoh_vs_wc0_half_T_deg"], 3),
      "| 이산 줄어든 각도 대 wc0*T/2", round(out["max_drop_diff_disc_vs_wc0_half_T_deg"], 3))
print("위상 여유 60도 이상인 가장 큰 T", out["largest_T_ms_pm_zoh_ge_60"], "ms | 50도 이상", out["largest_T_ms_pm_zoh_ge_50"], "ms")
print("최대 차이가 나는 T [ms]:", max(rows, key=lambda k: abs(rows[k]["drop_zoh_deg"] - rows[k]["wc0_times_half_T_deg"])),
      "| 이산 대 wc0*T/2:", max(rows, key=lambda k: abs(rows[k]["drop_disc_deg"] - rows[k]["wc0_times_half_T_deg"])),
      "| 이산 대 ZOH 루프:", max(rows, key=lambda k: abs(rows[k]["pm_disc_deg"] - rows[k]["pm_zoh_deg"])))
for t in (18, 19):
    print(t, "ms: ZOH 루프", round(rows[str(t)]["pm_zoh_deg"], 2), "| 이산", round(rows[str(t)]["pm_disc_deg"], 2))

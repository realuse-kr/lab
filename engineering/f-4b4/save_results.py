"""F-4B4: phase_margin_delay.py의 결과를 results_f4b4.json으로 저장하고, 다른 경로로 다시 구해 대조한다."""
import json
import numpy as np
from phase_margin_delay import (KP, KI, TAU, margin_cont, margin_disc, disc_pole_abs_max)

GRID_MS = list(range(1, 61))  # 지연 T를 1 ms부터 60 ms까지 1 ms 단위로 계산한다
TABLE_MS = [5, 10, 20, 40]

wc0, pm0 = margin_cont(0.0)
delay_margin_ms = float(np.radians(pm0) / wc0 * 1000)

cont = {"T_ms": GRID_MS, "wc_rad_per_s": [], "pm_deg": [], "drop_deg": [], "wc_times_T_deg": []}
for t_ms in GRID_MS:
    wc_t, pm = margin_cont(t_ms / 1000)
    cont["wc_rad_per_s"].append(wc_t)
    cont["pm_deg"].append(pm)
    cont["drop_deg"].append(pm0 - pm)
    cont["wc_times_T_deg"].append(float(np.degrees(wc0 * t_ms / 1000)))

disc = {}
for t_ms in GRID_MS[:40]:  # 이산 루프는 1 ms부터 40 ms까지
    T = t_ms / 1000
    w_no, pm_no = margin_disc(T, 0)
    w_d, pm_d = margin_disc(T, 1)
    w_2, pm_2 = margin_disc(T, 2)
    disc[str(t_ms)] = {
        "wc_no_delay": w_no, "pm_no_delay_deg": pm_no, "pm_one_sample_deg": pm_d, "pm_two_samples_deg": pm_2,
        "wc_times_T_deg": float(np.degrees(w_no * T)),
        "pole_abs_max_no_delay": disc_pole_abs_max(T, 0), "pole_abs_max_one_sample": disc_pole_abs_max(T, 1),
    }

first_unstable_one_sample = next((t for t in GRID_MS[:40] if disc[str(t)]["pole_abs_max_one_sample"] >= 1.0), None)
out = {
    "first_unstable_one_sample_ms": first_unstable_one_sample,
    "plant_tau_s": TAU, "kp": KP, "ki": KI,
    "continuous": {"wc_rad_per_s": wc0, "pm_no_delay_deg": pm0, "delay_margin_ms": delay_margin_ms, "grid": cont},
    "discrete": disc,
}

# 점검 1: 교차 각주파수를 |L(jw)| = 1의 닫힌 식으로 다시 구한다. TAU^2 w^4 + (1 - KP^2) w^2 - KI^2 = 0.
w2 = np.roots([TAU ** 2, 1 - KP ** 2, -KI ** 2])
wc_closed = float(np.sqrt(max(w2[np.isreal(w2)].real)))
out["check_crossover_matches_closed_form"] = bool(abs(wc_closed - wc0) < 1e-9)

# 점검 2: 지연 없는 위상 여유를 위상 식 atan(KP w/KI) - atan(TAU w)로 다시 구한다.
pm_formula = float(90.0 + np.degrees(np.arctan(KP * wc0 / KI)) - np.degrees(np.arctan(TAU * wc0)))
out["check_pm_matches_phase_formula"] = bool(abs(pm_formula - pm0) < 1e-6)

# 점검 3: 연속 시간 루프에서 위상 여유가 줄어든 각도가 모든 T에서 wc*T와 같다(주파수 응답에서 구한 값과 식의 값).
out["check_continuous_drop_equals_wc_T"] = bool(np.allclose(cont["drop_deg"], cont["wc_times_T_deg"], atol=1e-6))

# 점검 3b: 지연을 넣어도 교차 각주파수가 같다(지연의 크기가 모든 각주파수에서 1이다).
out["check_continuous_wc_unchanged"] = bool(np.allclose(cont["wc_rad_per_s"], wc0, atol=1e-9))

# 점검 4: 지연 마진 PM0/wc만큼 지연을 넣으면 위상 여유가 0이다.
out["check_delay_margin_gives_zero_pm"] = bool(abs(margin_cont(delay_margin_ms / 1000)[1]) < 1e-5)

# 점검 5: 이산 루프에서 한 샘플 지연이 줄인 위상 여유가 지연 없는 루프의 교차 각주파수 x T와 같다.
out["check_discrete_drop_equals_wc_T"] = bool(all(
    abs(((v["pm_no_delay_deg"] - v["pm_one_sample_deg"]) % 360) - v["wc_times_T_deg"]) < 1e-6 for v in disc.values()))

# 점검 6: 이산 폐루프가 안정(극점 크기 1 미만)인 것과 위상 여유가 양수인 것이 모든 T에서 같다.
out["check_stability_matches_pm_sign"] = bool(all(
    (v["pole_abs_max_one_sample"] < 1.0) == (v["pm_one_sample_deg"] > 0.0)
    and (v["pole_abs_max_no_delay"] < 1.0) == (v["pm_no_delay_deg"] > 0.0) for v in disc.values()))

# 점검 7: 두 샘플 지연은 위상 여유를 한 샘플 지연의 두 배만큼 줄인다.
out["check_two_samples_double_drop"] = bool(all(
    abs(((v["pm_no_delay_deg"] - v["pm_two_samples_deg"]) % 360) - 2 * v["wc_times_T_deg"]) < 1e-6 for v in disc.values()))

with open("results_f4b4.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k.startswith("check_")})
print("wc", round(wc0, 4), "pm0", round(pm0, 4), "delay_margin_ms", round(delay_margin_ms, 4))

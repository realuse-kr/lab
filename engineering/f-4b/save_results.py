"""F-4B 결과 저장: 샘플링 비율별 오버슈트·연속 시간 응답과의 최대 편차를 results_f4b.json으로 저장한다.
실행: python save_results.py (sweep_check.py와 같은 폴더에서, -I 없이)"""
import json
import numpy as np
import sweep_check as sc

FB = sc.bandwidth_hz()
RATIOS = [4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 40, 52, 64, 80, 104, 128, 160]

def spectral_radius(T, rule): # 폐루프 특성다항식 (z - 1)(z - a) + (1 - a)(c0 z + c1)의 근 중 가장 큰 크기
    a, (c0, c1) = np.exp(-T / sc.TAU), sc.rules(T)[rule]
    return float(max(abs(np.roots(np.polyadd(np.polymul([1, -1], [1, -a]), (1 - a) * np.array([c0, c1]))))))

def first_unstable_ratio(rule): # 비율을 8에서 0.5까지 0.01씩 내리며 처음 극점 크기가 1 이상이 되는 비율
    for r in np.arange(8.0, 0.5, -0.01):
        if spectral_radius(1 / (r * FB), rule) >= 1.0:
            return round(float(r), 2)

def row(r):
    T = 1 / (r * FB)
    out, oc = sc.metrics(T)
    return {"ratio": r, "T_ms": round(T * 1000, 3), "continuous_overshoot_pct": round(oc * 100, 3),
            "rules": {k: {"overshoot_pct": round(v[0] * 100, 3), "max_dev": round(float(v[1]), 5),
                          "max_pole_modulus": round(spectral_radius(T, k), 4)} for k, v in out.items()}}

sweep = [row(r) for r in RATIOS]
t20, oc20 = sc.metrics(0.02)
dev = {k: [sc.metrics(1 / (r * FB))[0][k][1] for r in (5, 10, 20, 40, 80, 160)] for k in sc.rules(0.02)}
res = {"fb_hz": round(FB, 4), "continuous_overshoot_pct": round(oc20 * 100, 3), "ratios": RATIOS, "sweep": sweep,
       "T20ms": {"ratio": round(1 / (0.02 * FB), 3), **{k: {"overshoot_pct": round(v[0] * 100, 3), "max_dev": round(float(v[1]), 5)} for k, v in t20.items()}},
       "first_unstable_ratio_scan_step_0.01": {k: first_unstable_ratio(k) for k in sc.rules(0.02)},
       "deviation_ratio_per_doubling_r5_to_160": {k: [round(v[i] / v[i + 1], 3) for i in range(5)] for k, v in dev.items()}}

# 연습문제 답: 상수를 바꾼 계산은 sweep_check의 상수를 직접 바꿔 다시 구한다
sc.KI = 20.0; sc.NUM, sc.DEN = [sc.KP, sc.KI], [sc.TAU, 1 + sc.KP, sc.KI]
fb2 = sc.bandwidth_hz(); o2, oc2 = sc.metrics(1 / (20 * fb2))
res["exercises"] = {"ex1_T20ms": res["T20ms"], "ex2_KI20": {"fb_hz": round(fb2, 4), "continuous_overshoot_pct": round(oc2 * 100, 3),
                    "ratio20_forward_euler_max_dev": round(float(o2["Forward Euler"][1]), 5)},
                    "ex3_forward_euler_dev_40_over_80": round(dev["Forward Euler"][3] / dev["Forward Euler"][4], 3)}
json.dump(res, open("results_f4b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res["first_unstable_ratio_scan_step_0.01"]), res["exercises"])

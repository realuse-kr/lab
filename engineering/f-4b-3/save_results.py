"""F-4B.3 결과 저장: 샘플링 비율별 연속 응답과의 편차·위상 여유, 허용 편차별 필요 비율, 출처 비율의 환산값을 results_f4b3.json으로 저장한다.
실행: python save_results.py (rule_multiple.py와 같은 폴더에서, -I 없이)"""
import json
import numpy as np
import rule_multiple as rm

FB = rm.bandwidth_hz()
W = np.logspace(0, 3, 400000)
L = (rm.KP * 1j * W + rm.KI) / (1j * W * (rm.TAU * 1j * W + 1)) # 연속 시간 개루프 L(s) = C(s)P(s)
i = np.argmax(np.abs(L) < 1)
WC, PMC = W[i], 180 + np.degrees(np.angle(L[i])) # 교차 주파수(rad/s)와 연속 시간 위상 여유
FC, FN = WC / (2 * np.pi), np.sqrt(rm.KI / rm.TAU) / (2 * np.pi) # 교차 주파수, 고유 주파수(Hz)
ZETA = (1 + rm.KP) / (2 * rm.TAU * np.sqrt(rm.KI / rm.TAU))
RULES = list(rm.rules(0.02))
MULTIPLES = [4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 52, 64, 80, 100, 128, 160]

def row(m):
    T = 1 / (m * FB)
    d = rm.deviation(T)
    pm = {k: rm.phase_margin(T, k) for k in RULES}
    return {"m": m, "T_ms": round(T * 1000, 2), "deviation": {k: round(float(v), 4) for k, v in d.items()},
            "dev_min": round(float(min(d.values())), 4), "dev_max": round(float(max(d.values())), 4),
            "pm_deg": {k: round(float(v), 1) for k, v in pm.items()},
            "pm_min": round(float(min(pm.values())), 1), "pm_max": round(float(max(pm.values())), 1),
            "pm_loss_deg": {k: round(float(PMC - v), 1) for k, v in pm.items()}, # 연속 시간 위상 여유에서 줄어든 값
            "zoh_delay_loss_deg": round(float(np.degrees(WC * T / 2)), 1)} # ZOH를 T/2 지연으로 근사한 위상 손실 ωc·T/2

def crossing(level, key): # key(m)이 level과 같아지는 비율(이분법). 격자마다 단조 감소를 확인한 뒤 구한다
    grid = np.arange(4.0, 160.01, 0.5)
    d = [key(m) for m in grid]
    assert all(d[j] >= d[j + 1] for j in range(len(d) - 1)), "편차가 비율에 따라 단조 감소하지 않는다"
    lo, hi = 4.0, 160.0
    assert key(lo) > level >= key(hi)
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if key(mid) > level else (lo, mid)
    return round(hi, 1)

def pm_cross(loss, pick): # 위상 손실이 loss도 이내가 되는 최소 비율(이분법). pick은 규칙 선택 함수(min=가장 나쁜 규칙, max=가장 좋은 규칙)
    f = lambda m: pick(rm.phase_margin(1 / (m * FB), k) for k in RULES) - (PMC - loss)
    grid = np.arange(4.0, 160.01, 4.0) # 위상 여유 계산이 느려 4 간격 격자로 단조 증가를 확인한다
    v = [f(m) for m in grid]
    assert all(v[j] <= v[j + 1] for j in range(len(v) - 1)), "위상 여유가 비율에 따라 단조 증가하지 않는다"
    lo, hi = 4.0, 160.0
    assert f(lo) < 0 <= f(hi)
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) >= 0 else (mid, hi)
    return round(hi, 1)

worst = lambda m: max(rm.deviation(1 / (m * FB)).values()) # 세 규칙 중 가장 큰 편차
best = lambda m: min(rm.deviation(1 / (m * FB)).values())  # 세 규칙 중 가장 작은 편차
k_ac = FC / FB # 교차 주파수 ÷ 대역폭: 교차 주파수 기준 비율에 곱하면 대역폭 기준 비율이 된다
sources = { # 출처가 말한 비율을 이 조건의 대역폭 기준 비율(m)로 읽은 값. Årzén은 교차 주파수 기준 ωc·h를 환산했다(a: 0.15~0.5, b: 0.05~0.14)
    "Goodwin": [10, 10], "Franklin": [20, None], "Plett": [30, 30], "Melchiorri": [6, 20], "Hirata_Powell": [8, 40],
    "Bouvet": [6, 25], "Arzen_a": [round(float(2 * np.pi / 0.5 * k_ac), 1), round(float(2 * np.pi / 0.15 * k_ac), 1)],
    "Arzen_b": [round(float(2 * np.pi / 0.14 * k_ac), 1), round(float(2 * np.pi / 0.05 * k_ac), 1)]}
src_rows = {k: [row(v) for v in vs if v is not None] for k, vs in sources.items()}
res = {"fb_hz": round(float(FB), 4), "wb_rad_s": round(float(2 * np.pi * FB), 2), "wc_rad_s": round(float(WC), 3), "fc_hz": round(float(FC), 4),
       "fn_hz": round(float(FN), 4), "zeta": round(float(ZETA), 3), "pm_continuous_deg": round(float(PMC), 2),
       "fc_over_fb": round(float(k_ac), 4), "fn_over_fb": round(float(FN / FB), 4), "multiples": MULTIPLES,
       "sweep": [row(m) for m in MULTIPLES], "sources_m": sources, "sources_rows": src_rows,
       "tolerance_multiple_worst_rule": {str(L_): crossing(L_, worst) for L_ in (0.05, 0.02, 0.01)},
       "tolerance_multiple_best_rule": {str(L_): crossing(L_, best) for L_ in (0.05, 0.02, 0.01)},
       "pm_loss_multiple_worst_rule": {str(x): pm_cross(x, min) for x in (15, 10, 5)},
       "pm_loss_multiple_best_rule": {str(x): pm_cross(x, max) for x in (15, 10, 5)},
       "same_T_denominators": {"T_ms": round(1000 / (10 * FB), 2), "fs_over_fb": 10.0, "fs_over_fc": round(float(10 * FB / FC), 2), "fs_over_fn": round(float(10 * FB / FN), 2)}}
# 연습문제 답: 상수를 바꾼 계산은 rule_multiple의 상수를 직접 바꿔 다시 구한다
T20 = 0.02
res["exercises"] = {"ex1_T20ms": {"m": round(float(1 / (T20 * FB)), 3), "dev": {k: round(float(v), 4) for k, v in rm.deviation(T20).items()}, "tustin_pm_deg": round(float(rm.phase_margin(T20, "Tustin")), 1)}}
rm.KI = 20.0
rm.NUM, rm.DEN = [rm.KP, rm.KI], [rm.TAU, 1 + rm.KP, rm.KI]
fb2 = rm.bandwidth_hz()
res["exercises"]["ex2_KI20"] = {"fb_hz": round(float(fb2), 4), "dev_at_m20": {k: round(float(v), 4) for k, v in rm.deviation(1 / (20 * fb2)).items()}}
json.dump(res, open("results_f4b3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["fb_hz"], res["fc_hz"], res["fn_hz"], res["zeta"], res["pm_continuous_deg"], res["fc_over_fb"], res["fn_over_fb"])
print(res["tolerance_multiple_worst_rule"], res["tolerance_multiple_best_rule"], res["same_T_denominators"], res["exercises"], res["sources_m"])

"""F-4B.2 결과 저장: 세 규칙의 규칙 간 최대 차이와 극점 크기를 results_f4b2.json으로 저장한다.
실행: python save_results.py (rule_gap.py와 같은 폴더에서, -I 없이)"""
import json
import numpy as np
import rule_gap as rg

FB = rg.bandwidth_hz()
RATIOS = [4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 40, 52, 64, 80, 104, 128, 160]
RULES = list(rg.rules(0.02))

def gap_at(r):
    return float(rg.max_gap(1 / (r * FB)))

def crossing_ratio(level): # 규칙 간 최대 차이가 level과 같아지는 비율(이분법). 먼저 비율 4~160에서 차이가 0.5 간격 격자마다 줄어드는지 확인한다
    grid = np.arange(4.0, 160.01, 0.5)
    d = [gap_at(r) for r in grid]
    assert all(d[i] >= d[i + 1] for i in range(len(d) - 1)), "차이가 비율에 따라 단조 감소하지 않는다"
    lo, hi = 4.0, 160.0
    assert gap_at(lo) > level >= gap_at(hi)
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap_at(mid) > level else (lo, mid)
    return round(hi, 2)

def unstable(r, rule):
    return bool(rg.pole_modulus(1 / (r * FB), rule) >= 1.0)

scan = [round(float(r), 2) for r in np.arange(8.0, 0.499, -0.01)]
verdict = {r: tuple(unstable(r, k) for k in RULES) for r in scan}
first = {k: next(r for r in scan if verdict[r][i]) for i, k in enumerate(RULES)}
for i, k in enumerate(RULES): # 처음 불안정해진 비율 아래에서는 0.5까지 계속 불안정한지 확인
    assert all(verdict[r][i] for r in scan if r <= first[k]) and not any(verdict[r][i] for r in scan if r > first[k])
mixed = [r for r in scan if len(set(verdict[r])) == 2]

def row(r):
    T = 1 / (r * FB)
    return {"ratio": r, "T_ms": round(T * 1000, 3), "max_gap": round(gap_at(r), 5),
            "pole_modulus": {k: round(float(rg.pole_modulus(T, k)), 4) for k in RULES}}

sweep = [row(r) for r in RATIOS]
g = {r: gap_at(r) for r in (5, 10, 20, 40, 80, 160)}
crossing = {L: crossing_ratio(L) for L in (0.05, 0.02, 0.01)} # 규칙 간 최대 차이가 목표값의 5%, 2%, 1%가 되는 비율
res = {"fb_hz": round(FB, 4), "ratios": RATIOS, "sweep": sweep,
       "gap_ratio_per_doubling_r5_to_160": {f"{b}->{a}": round(g[b] / g[a], 4) for b, a in ((5, 10), (10, 20), (20, 40), (40, 80), (80, 160))},
       "gap_levels": {str(L): {"ratio": crossing[L], "T_ms": round(float(1000 / (crossing[L] * FB)), 2)} for L in crossing},
       "first_unstable_ratio_scan_step_0.01": first,
       "mixed_verdict_ratio_range": [min(mixed), max(mixed)],
       "verdict_samples": {str(r): dict(zip(RULES, ("불안정" if u else "안정" for u in verdict[r]))) for r in (4.0, 3.5, 3.0, 2.5, 2.0, 1.5)}}

# 연습문제 답: 상수를 바꾼 계산은 rule_gap의 상수를 직접 바꿔 다시 구한다
res["exercises"] = {"ex1_T20ms": {"ratio": round(float(1 / (0.02 * FB)), 3), "max_gap": round(float(rg.max_gap(0.02)), 5)}}
rg.KI = 20.0; rg.NUM, rg.DEN = [rg.KP, rg.KI], [rg.TAU, 1 + rg.KP, rg.KI]
fb2 = rg.bandwidth_hz()
res["exercises"]["ex2_KI20"] = {"fb_hz": round(float(fb2), 4), "ratio20_max_gap": round(float(rg.max_gap(1 / (20 * fb2))), 5)}
res["exercises"]["ex3_gap_40_over_80"] = round(g[40] / g[80], 4)
json.dump(res, open("results_f4b2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["gap_levels"], res["first_unstable_ratio_scan_step_0.01"], res["mixed_verdict_ratio_range"], res["exercises"])

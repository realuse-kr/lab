"""F-4B2.3 결과 저장: 프리워핑 Tustin 계수, 주파수 응답비, 스텝·램프 입력의 제어기 출력 차이, 폐루프 편차를 results_f4b23.json으로 저장한다.
실행: python save_results.py (prewarp_tustin.py와 같은 폴더에서)"""
import json
import numpy as np
from scipy import signal
import prewarp_tustin as pt

KP, KI, TAU, TEND = pt.KP, pt.KI, 0.1, 2.0  # 플랜트 P(s) = 1/(TAU*s + 1), 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI]  # 연속 시간 폐루프 전달함수
W = np.logspace(0, 3, 400000)
FB = W[np.argmax(np.abs(signal.freqs(NUM, DEN, W)[1]) < 1 / np.sqrt(2))] / (2 * np.pi)  # 폐루프 -3 dB 대역폭(Hz)
L = (KP * 1j * W + KI) / (1j * W * (TAU * 1j * W + 1))  # 연속 시간 개루프
WC = round(float(W[np.argmax(np.abs(L) < 1)]), 3)  # 교차 주파수(rad/s), 소수 셋째 자리로 둔다
W2 = 60.0  # 비교용 두 번째 기준 주파수(rad/s)
T20 = 0.02

def rules(T):
    return {"Tustin": pt.tustin(T), "프리워핑 ωc": pt.prewarped(T, WC), "프리워핑 60": pt.prewarped(T, W2)}

def deviation(T, c):  # 이산 폐루프 스텝 응답과 연속 시간 폐루프 스텝 응답의 샘플 시점별 차이 중 가장 큰 절댓값
    n, a = int(round(TEND / T)), np.exp(-T / TAU)
    yc = signal.step((NUM, DEN), T=np.arange(n + 1) * T)[1]
    y, e_old, u_old = [0.0], 0.0, 0.0
    for _ in range(n):
        e = 1.0 - y[-1]
        u_old = c[0] * e + c[1] * e_old + u_old
        e_old = e
        y.append(a * y[-1] + (1 - a) * u_old)
    return float(np.abs(np.array(y) - yc).max())

res = {"fb_hz": round(float(FB), 4), "wc_rad_s": round(WC, 4), "w2_rad_s": W2, "T20_ms": 20.0, "m_at_T20": round(float(1 / (T20 * FB)), 3),
       "nyquist_rad_s_T20": round(float(np.pi / T20), 3), "g_wc_T20": round(float(pt.warp_gain(T20, WC)), 6), "g_60_T20": round(float(pt.warp_gain(T20, W2)), 6)}
t = np.arange(int(round(TEND / T20)) + 1) * T20
coef, ctrl = {}, {}
for name, c in rules(T20).items():
    step = pt.controller_out(c, np.ones(t.size)) - (KP + KI * t)
    ramp = pt.controller_out(c, t) - (KP * t + KI * t ** 2 / 2)
    coef[name] = [round(float(c[0]), 6), round(float(c[1]), 6)]
    ctrl[name] = {"step_diff_first_sample": round(float(step[0]), 6), "zero_z": round(float(-c[1] / c[0]), 6), "low_freq_gain_c0_plus_c1_over_T": round(float((c[0] + c[1]) / T20), 6),
                  "step_diff_at_2s": round(float(step[-1]), 6), "ramp_diff_at_2s": round(float(ramp[-1]), 6)}
res["coef_T20"], res["controller_T20"] = coef, ctrl
res["check_substitute_max_abs_diff"] = float(max(np.abs(np.array(pt.prewarped(T20, w)) - np.array(pt.substitute(T20, w))).max() for w in (WC, W2)))
ws = [5.0, 12.0, WC, 60.0, 100.0]  # 주파수 응답비를 보는 각주파수(rad/s)
res["freq_ratio_T20"] = {name: [{"w": w, "mag": round(float(abs(pt.ratio(c, T20, w))), 5), "phase_deg": round(float(np.degrees(np.angle(pt.ratio(c, T20, w)))), 4)} for w in ws]
                         for name, c in rules(T20).items()}
res["match_at_w0"] = {"프리워핑 ωc": [round(float(abs(pt.ratio(pt.prewarped(T20, WC), T20, WC))), 12), round(float(np.degrees(np.angle(pt.ratio(pt.prewarped(T20, WC), T20, WC)))), 12)],
                      "프리워핑 60": [round(float(abs(pt.ratio(pt.prewarped(T20, W2), T20, W2))), 12), round(float(np.degrees(np.angle(pt.ratio(pt.prewarped(T20, W2), T20, W2)))), 12)]}
ms = [6, 8, 10, 20, 25, 30, 40, 100]
res["multiples"] = ms
res["sweep"] = [{"m": m, "T_ms": round(float(1000 / (m * FB)), 2), "g_wc": round(float(pt.warp_gain(1 / (m * FB), WC)), 6),
                 "deviation": {k: round(deviation(1 / (m * FB), c), 6) for k, c in rules(1 / (m * FB)).items()}} for m in ms]
def ramp_at_2s(T, c):
    tt = np.arange(int(round(TEND / T)) + 1) * T
    return float((pt.controller_out(c, tt) - (KP * tt + KI * tt ** 2 / 2))[-1])
res["exercises"] = {"ex1_T5ms_wc_coef": [round(float(v), 4) for v in pt.prewarped(0.005, WC)],
                    "ex2_w0_100_T20ms_coef": [round(float(v), 4) for v in pt.prewarped(0.02, 100.0)],
                    "ex3_T10ms_wc_ramp_diff_at_2s": round(ramp_at_2s(0.01, pt.prewarped(0.01, WC)), 4)}
json.dump(res, open("results_f4b23.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["fb_hz"], res["wc_rad_s"], res["m_at_T20"], res["nyquist_rad_s_T20"], res["g_wc_T20"], res["g_60_T20"], res["check_substitute_max_abs_diff"])
print(res["coef_T20"])
for k, v in res["controller_T20"].items(): print(k, v)
for k, v in res["freq_ratio_T20"].items(): print(k, v)
print(res["match_at_w0"])
for r in res["sweep"]: print(r)
print(res["exercises"])

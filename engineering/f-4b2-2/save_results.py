"""F-4B2.2 결과 저장: PI 제어기의 극·영점 매칭 계수, 스텝·램프·지수 입력의 제어기 출력 차이, ωc 응답, 폐루프 편차를 results_f4b22.json으로 저장한다.
실행: python save_results.py (matched_pz.py와 같은 폴더에서)"""
import json
import numpy as np
from scipy import signal
import matched_pz as mp

KP, KI, A, TAU, TEND = mp.KP, mp.KI, mp.A, 0.1, 2.0   # 플랜트 P(s) = 1/(TAU*s + 1), 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI]                # 연속 시간 폐루프 전달함수
W = np.logspace(0, 3, 400000)
FB = W[np.argmax(np.abs(signal.freqs(NUM, DEN, W)[1]) < 1 / np.sqrt(2))] / (2 * np.pi)   # 폐루프 -3 dB 대역폭(Hz)
L = (KP * 1j * W + KI) / (1j * W * (TAU * 1j * W + 1))                                   # 연속 시간 개루프
WC = W[np.argmax(np.abs(L) < 1)]                                                         # 교차 주파수(rad/s)
T20 = 0.02

def deviation(T, c):   # 이산 폐루프 스텝 응답과 연속 시간 폐루프 스텝 응답의 샘플 시점별 차이 중 가장 큰 절댓값
    n, a = int(round(TEND / T)), np.exp(-T / TAU)
    yc = signal.step((NUM, DEN), T=np.arange(n + 1) * T)[1]
    y, e_old, u_old = [0.0], 0.0, 0.0
    for _ in range(n):
        e = 1.0 - y[-1]
        u_old = c[0] * e + c[1] * e_old + u_old
        e_old = e
        y.append(a * y[-1] + (1 - a) * u_old)
    return float(np.abs(np.array(y) - yc).max())

t = np.arange(int(round(TEND / T20)) + 1) * T20
res = {"fb_hz": round(float(FB), 4), "wc_rad_s": round(float(WC), 3), "T20_ms": 20.0, "m_at_T20": round(float(1 / (T20 * FB)), 3), "zero_s_rad_s": -A}
coef, ctrl, freq = {}, {}, {}
for name, c in mp.rules(T20).items():
    step = mp.controller_out(c, np.ones(t.size)) - (KP + KI * t)
    ramp = mp.controller_out(c, t) - (KP * t + KI * t ** 2 / 2)
    expo = mp.controller_out(c, np.exp(-A * t)) - KP   # 연속 시간 제어기에 e(t) = exp(-A t)를 넣으면 출력은 KP로 일정하다
    z = np.exp(1j * WC * T20)
    ratio = (c[0] * z + c[1]) / (z - 1) / (KP + KI / (1j * WC))
    coef[name] = [round(float(c[0]), 6), round(float(c[1]), 6)]
    ctrl[name] = {"zero_z": round(float(-c[1] / c[0]), 6), "low_freq_gain_c0_plus_c1_over_T": round(float((c[0] + c[1]) / T20), 6),
                  "step_diff_at_2s": round(float(step[-1]), 6), "ramp_diff_at_2s": round(float(ramp[-1]), 6),
                  "exp_diff_min": round(float(expo.min()), 6), "exp_diff_max": round(float(expo.max()), 6)}
    freq[name] = {"gain_ratio": round(float(abs(ratio)), 5), "phase_diff_deg": round(float(np.degrees(np.angle(ratio))), 4)}
res["coef_T20"], res["controller_T20"], res["freq_at_wc_T20"] = coef, ctrl, freq
p, k = np.exp(-A * T20), mp.matched(T20)[0]
x = A * T20
res["analytic_T20"] = {"zero_z": float(p), "k": float(k), "offset_vs_kp": float(k - KP),
                       "ramp_slope": float(KI * T20 * (1 / (1 - p) - 1 / x - 0.5)), "ramp_diff_at_2s": float(2 * KI * T20 * (1 / (1 - p) - 1 / x - 0.5)),
                       "ramp_slope_small_x_approx": float(KI * A * T20 ** 2 / 12)}
ms = [6, 8, 10, 20, 25, 30, 40, 100]
res["multiples"] = ms
res["sweep"] = [{"m": m, "T_ms": round(float(1000 / (m * FB)), 2), "deviation": {k_: round(deviation(1 / (m * FB), c), 6) for k_, c in mp.rules(1 / (m * FB)).items()}} for m in ms]
def ramp_at_2s(T, rule):
    tt = np.arange(int(round(TEND / T)) + 1) * T
    return float((mp.controller_out(mp.rules(T)[rule], tt) - (KP * tt + KI * tt ** 2 / 2))[-1])
res["exercises"] = {"ex1_T5ms_coef": [round(float(v), 4) for v in mp.matched(0.005)],
                    "ex3_T10ms_ramp_diff_at_2s": round(ramp_at_2s(0.01, "극·영점 매칭"), 4)}
mp.KI = 20.0; mp.A = mp.KI / mp.KP   # 연습문제 2: Ki를 20으로 바꾸면 영점이 s = -10으로 옮겨진다 (matched가 모듈 상수를 읽는다)
res["exercises"]["ex2_KI20_T20ms_coef"] = [round(float(v), 4) for v in mp.matched(0.02)]
json.dump(res, open("results_f4b22.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["fb_hz"], res["wc_rad_s"], res["m_at_T20"]); print(res["coef_T20"])
for kx, v in res["controller_T20"].items(): print(kx, v)
print(res["freq_at_wc_T20"]); print(res["analytic_T20"])
for r in res["sweep"]: print(r)
print(res["exercises"])

"""F-4B2 결과 저장: PI 제어기의 스텝 불변 이산화 계수, 스텝·램프·구간 상수 입력과 ωc 응답 비교, 폐루프 편차를 results_f4b2.json으로 저장한다.
실행: python save_results.py (step_invariant.py와 같은 폴더에서, -I 없이)"""
import json
import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
import step_invariant as si

KP, KI, TAU, TEND = si.KP, si.KI, 0.1, 2.0 # 플랜트 P(s) = 1/(TAU*s + 1), 계산 구간 0~TEND초
NUM, DEN = [KP, KI], [TAU, 1 + KP, KI] # 연속 시간 폐루프 전달함수
W = np.logspace(0, 3, 400000)
FB = W[np.argmax(np.abs(signal.freqs(NUM, DEN, W)[1]) < 1 / np.sqrt(2))] / (2 * np.pi) # 폐루프 -3 dB 대역폭(Hz)
L = (KP * 1j * W + KI) / (1j * W * (TAU * 1j * W + 1)) # 연속 시간 개루프
WC = W[np.argmax(np.abs(L) < 1)] # 교차 주파수(rad/s)
RULES = list(si.rules(0.02))
T20 = 0.02

def deviation(T, c): # 이산 폐루프 스텝 응답과 연속 시간 폐루프 스텝 응답의 샘플 시점별 차이 중 가장 큰 절댓값
    n, a = int(round(TEND / T)), np.exp(-T / TAU)
    yc = signal.step((NUM, DEN), T=np.arange(n + 1) * T)[1]
    y, e_old, u_old = [0.0], 0.0, 0.0
    for _ in range(n):
        e = 1.0 - y[-1]
        u_old = c[0] * e + c[1] * e_old + u_old
        e_old = e
        y.append(a * y[-1] + (1 - a) * u_old)
    return float(np.abs(np.array(y) - yc).max())

def ct_with_held_input(e): # 연속 시간 PI 제어기에 샘플 사이 일정한 입력 e(t) = e[k]를 넣고 샘플 시점 출력을 수치 적분으로 구한다
    x, out = 0.0, []
    for ek in e:
        out.append(KP * ek + KI * x) # u(kT) = KP*e + KI*(e의 적분)
        x = solve_ivp(lambda t, s: [ek], (0.0, T20), [x], rtol=1e-12, atol=1e-14).y[0, -1]
    return np.array(out)

t = np.arange(int(round(TEND / T20)) + 1) * T20
res = {"fb_hz": round(float(FB), 4), "wc_rad_s": round(float(WC), 3), "T20_ms": 20.0, "m_at_T20": round(float(1 / (T20 * FB)), 3)}
coef, ctrl, freq = {}, {}, {}
for name, c in si.rules(T20).items():
    coef[name] = [round(float(c[0]), 6), round(float(c[1]), 6)]
    step = si.controller_out(c, np.ones(t.size)) - (KP + KI * t)
    ramp = si.controller_out(c, t) - (KP * t + KI * t ** 2 / 2)
    z = np.exp(1j * WC * T20)
    ratio = (c[0] * z + c[1]) / (z - 1) / (KP + KI / (1j * WC)) # 이산 제어기 응답 ÷ 연속 시간 제어기 응답(ωc)
    ctrl[name] = {"step_diff_min": round(float(step.min()), 6), "step_diff_max": round(float(step.max()), 6),
                  "ramp_diff_at_2s": round(float(ramp[-1]), 6), "ramp_diff_over_t_per_s": round(float(ramp[-1] / TEND), 6)}
    freq[name] = {"gain_ratio": round(float(abs(ratio)), 4), "phase_diff_deg": round(float(np.degrees(np.angle(ratio))), 2)}
res["coef_T20"], res["controller_T20"], res["freq_at_wc_T20"] = coef, ctrl, freq
res["analytic_T20"] = {"ramp_slope_step_invariant": -KI * T20 / 2, "step_offset_backward": KI * T20, "step_offset_tustin": KI * T20 / 2,
                       "ramp_slope_backward": KI * T20 / 2, "ramp_slope_tustin": 0.0}
res["si_equals_forward_euler_max_coef_diff"] = float(max(abs(a - b) for T in (0.001, 0.005, 0.02, 0.05, 0.1)
                                                         for a, b in zip(si.step_invariant(T), si.rules(T)["Forward Euler"])))
rng = np.random.default_rng(0)
e = rng.uniform(-1, 1, t.size) # 샘플 사이 일정한 난수 입력(길이 101)
u_ct = ct_with_held_input(e)
res["held_random_input_max_diff"] = {k: float(np.abs(si.controller_out(c, e) - u_ct).max()) for k, c in si.rules(T20).items()}
ms = [6, 8, 10, 20, 25, 30, 40, 100]
res["multiples"] = ms
res["sweep"] = [{"m": m, "T_ms": round(1000 / (m * FB), 2), "deviation": {k: round(deviation(1 / (m * FB), c), 4) for k, c in si.rules(1 / (m * FB)).items()}} for m in ms]
res["exercises"] = {"ex1_T10ms": {k: {"ramp_diff_at_2s": round(float((si.controller_out(c, np.arange(201) * 0.01) - (KP * np.arange(201) * 0.01 + KI * (np.arange(201) * 0.01) ** 2 / 2))[-1]), 4)}
                                  for k, c in si.rules(0.01).items()}}
tt = np.arange(int(round(TEND / 0.0025)) + 1) * 0.0025 # 연습문제 3: T = 2.5 ms에서 스텝 불변 제어기의 램프 입력 차이
res["exercises"]["ex3_T2p5ms"] = {"ramp_diff_at_2s": round(float((si.controller_out(si.step_invariant(0.0025), tt) - (KP * tt + KI * tt ** 2 / 2))[-1]), 4)}
si.KI = 20.0 # 연습문제 2: Ki를 20으로 바꾸면 스텝 불변 계수(c0, c1)는? (step_invariant가 모듈 상수를 읽는다)
res["exercises"]["ex2_KI20_T20ms"] = {"coef": [round(float(v), 4) for v in si.step_invariant(0.02)]}
json.dump(res, open("results_f4b2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(res["fb_hz"], res["wc_rad_s"], res["m_at_T20"], res["coef_T20"])
print(res["controller_T20"]); print(res["freq_at_wc_T20"]); print(res["analytic_T20"])
print(res["si_equals_forward_euler_max_coef_diff"], res["held_random_input_max_diff"])
for r in res["sweep"]: print(r)
print(res["exercises"])
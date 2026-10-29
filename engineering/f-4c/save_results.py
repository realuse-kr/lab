import json
import numpy as np
from scipy.signal import cont2discrete, dlsim, dlti
from pi_discretization import METHODS, OS_CONT, KP, KI, TAU, run

TS = (0.005, 0.01, 0.02, 0.04, 0.05)
out = {"os_cont": OS_CONT, "table": {}, "checks": {}}
for T in TS:
    out["table"][str(T)] = {n: dict(zip(("os", "maxz"), run(k, T))) for n, k in METHODS.items()}

def closed_loop(method, T):  # 폐루프 전달함수 L/(1+L)의 분자·분모
    num, _, _ = cont2discrete(([KP, KI], [1, 0]), T, method=method)
    b = np.squeeze(num); a = np.exp(-T / TAU)
    den = np.polyadd(np.polymul([1, -a], [1, -1]), (1 - a) * b)
    return (1 - a) * b, den

def os_samples(method, T):  # 샘플 시각만 본 오버슈트: dlsim과 직접 비교
    n, d = closed_loop(method, T)
    k = round(1.5 / T)
    _, y = dlsim(dlti(n, d, dt=T), np.ones(k))
    return (y.max() - 1) * 100

def os_samples_direct(method, T):
    num, _, _ = cont2discrete(([KP, KI], [1, 0]), T, method=method)
    b0, b1 = np.squeeze(num); a = np.exp(-T / TAU)
    y = u = e_old = 0.0; pk = -9
    for _ in range(round(1.5 / T)):
        e = 1 - y; u += b0 * e + b1 * e_old
        y, e_old = a * y + (1 - a) * u, e
        pk = max(pk, y)  # y[k+1]
    return (pk - 1) * 100

c = out["checks"]
c["dlsim_vs_loop_maxdiff"] = max(abs(os_samples(k, T) - os_samples_direct(k, T)) for k in METHODS.values() for T in TS)
c["numerators"] = {n: [float(x) for x in np.squeeze(cont2discrete(([KP, KI], [1, 0]), 0.02, method=k)[0])] for n, k in METHODS.items()}
T = 0.02
c["closed_form_numerators"] = {"Forward Euler": [KP, KI * T - KP], "Backward Euler": [KP + KI * T, -KP], "Tustin": [KP + KI * T / 2, KI * T / 2 - KP]}
c["small_T_gap_pp"] = {n: run(k, 2e-4)[0] - OS_CONT for n, k in METHODS.items()}
def os_fine(method, T, m=200):  # 샘플 사이를 m등분해 직접 확인
    num, _, _ = cont2discrete(([KP, KI], [1, 0]), T, method=method)
    b0, b1 = np.squeeze(num); a = np.exp(-T / TAU); aj = np.exp(-np.arange(1, m + 1) * T / (m * TAU))
    y = u = e_old = pk = 0.0
    for _ in range(round(1.5 / T)):
        e = 1 - y; u += b0 * e + b1 * e_old
        pk = max(pk, (aj * y + (1 - aj) * u).max()); y, e_old = a * y + (1 - a) * u, e
    return (pk - 1) * 100
c["intersample_maxdiff"] = max(abs(run(k, T)[0] - os_fine(k, T)) for k in METHODS.values() for T in TS)
c["all_stable"] = all(v["maxz"] < 1 for r in out["table"].values() for v in r.values())
c["rank_ok_T<=40ms"] = all(r["Backward Euler"]["os"] < r["Tustin"]["os"] < r["Forward Euler"]["os"] for t, r in out["table"].items() if float(t) <= 0.04)
r50 = out["table"]["0.05"]
c["T50_flip"] = r50["Backward Euler"]["os"] > r50["Forward Euler"]["os"]
def first_unstable(k):
    for T in np.arange(0.005, 0.2, 0.001):
        if run(k, T)[1] >= 1: return round(float(T), 3)
c["first_unstable_T"] = {n: first_unstable(k) for n, k in METHODS.items()}
def last_ok(k, lim=1.0):  # 1 ms 간격으로 훑어 오버슈트 오차가 lim %p 이하인 마지막 주기(처음 넘기면 중단)
    best = None
    for T in np.arange(0.001, 0.08, 0.001):
        if run(k, T)[0] - OS_CONT > lim: break
        best = round(float(T), 3)
    return best
c["max_T_error_le_1pp"] = {n: last_ok(k) for n, k in METHODS.items()}
def first_above(ka, kb):  # ka의 오버슈트가 kb보다 처음 커지는 주기(1 ms 간격, 40~60 ms)
    for T in np.arange(0.040, 0.0601, 0.001):
        if run(ka, T)[0] > run(kb, T)[0]: return round(float(T), 3)
c["first_T_backward_above_tustin"] = first_above("backward_diff", "bilinear")
c["first_T_backward_above_forward"] = first_above("backward_diff", "euler")
c["backward_error_pp_22_28ms"] = {str(round(float(T), 3)): round(run("backward_diff", T)[0] - OS_CONT, 3) for T in np.arange(0.022, 0.0281, 0.001)}  # 1%p 근처에서 오차가 주기에 단조 증가하지 않음을 확인
json.dump(out, open("results_f4c.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item())
print(json.dumps(c, ensure_ascii=False, indent=1, default=lambda o: o.item()))

"""F-4D.3 결과 저장: delay_check.py의 함수로 본문 표의 값, 검증 계산, 연습문제 답을 results_f4d3.json에 쓴다."""
import json, sys
import numpy as np
import scipy
from scipy.signal import cont2discrete, dstep

import delay_check as dc

SRC = open("delay_check.py", encoding="utf-8").read()


def load(T=0.02, KI=40.0):
    """delay_check.py의 상수 줄을 바꿔 다른 주기·적분 게인으로 실행하는 사본을 만든다."""
    ns = {"__name__": "copy"}
    exec(SRC.replace("KP, KI, T, TAU = 2.0, 40.0, 0.02, 0.1", f"KP, KI, T, TAU = 2.0, {KI}, {T}, 0.1"), ns)
    return ns


def cplx(z):
    return [round(float(z.real), 6), round(float(z.imag), 6)]


def order_residuals(mod, rule, delay, orders=(1, 2, 3, 4)):
    """차수 n별 맞춤 잔차. read_poles와 같은 행 구성을 쓴다."""
    h = np.diff(np.r_[0.0, mod["step_response"](rule, delay)])
    out = {}
    for n in orders:
        A = np.array([[-h[k - j] for j in range(1, n + 1)] for k in range(n + 3, len(h))])
        p = np.linalg.lstsq(A, h[n + 3:], rcond=None)[0]
        out[str(n)] = float(np.linalg.norm(A @ p - h[n + 3:]))
    return out


def case(mod, rule, delay):
    y = mod["step_response"](rule, delay)
    n, p = mod["read_poles"](y)
    r = mod["char_poles"](rule, delay)
    return {"y1": float(y[1]), "y2": float(y[2]), "read_order": n,
            "read_poles": [cplx(z) for z in sorted(p, key=lambda z: (round(z.real, 9), z.imag))],
            "char_poles": [cplx(z) for z in sorted(r, key=lambda z: (round(z.real, 9), z.imag))],
            "max_gap": float(max(min(abs(p - q)) for q in r)), "max_modulus": float(max(abs(r))),
            "y_last": float(y[-1])}


res = {"env": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
       "params": {"KP": dc.KP, "KI": dc.KI, "T": dc.T, "TAU": dc.TAU, "samples": 60}}

# 1) 닫힌 형태 계수 = SciPy cont2discrete 계수 (c0, c1, d1 = -1)
methods = {"ZOH, Forward Euler": ("zoh", "euler"), "Backward Euler": ("backward_diff",), "Tustin": ("bilinear",)}
diffs = []
for rule, ms in methods.items():
    for m in ms:
        num, den, _ = cont2discrete(([dc.KP, dc.KI], [1.0, 0.0]), dc.T, method=m)
        c0, c1, d1 = np.squeeze(num)[0], np.squeeze(num)[1], np.squeeze(den)[1]
        diffs.append(max(abs(c0 - dc.RULES[rule][0]), abs(c1 - dc.RULES[rule][1]), abs(d1 + 1.0)))
res["coefficient_vs_cont2discrete_max_abs_diff"] = float(max(diffs))

# 2) 본문 표: 규칙 x 지연
res["cases"] = {f"{rule}|delay{d}": case(dc.__dict__, rule, d) for rule in dc.RULES for d in (0, 1)}
res["order_residuals_ZOH_FE"] = {f"delay{d}": order_residuals(dc.__dict__, "ZOH, Forward Euler", d) for d in (0, 1)}

# 3) 코드 응답 = 폐루프 전달함수 스텝 응답 (경로 A와 B의 응답 일치)
a = np.exp(-dc.T / dc.TAU)
tf_diff = {}
for rule in dc.RULES:
    for d in (0, 1):
        c0, c1 = dc.RULES[rule]
        den = np.polyadd(np.polymul(np.polymul([1, -1], [1, -a]), np.r_[1.0, np.zeros(d)]), (1 - a) * np.array([c0, c1]))
        num = (1 - a) * np.array([c0, c1])
        _, (ys,) = dstep((num, den, dc.T), n=61)
        tf_diff[f"{rule}|delay{d}"] = float(np.max(abs(ys.ravel() - dc.step_response(rule, d, 60))))
res["code_vs_closed_loop_tf_step_max_abs_diff"] = tf_diff

# 4) 주기별 가장 큰 극점 크기와 읽은 극점 수
sweep = {}
for T in (0.01, 0.02, 0.04):
    mod = load(T=T)
    sweep[f"T{int(T * 1000)}ms"] = {f"{rule}|delay{d}": {k: v for k, v in case(mod, rule, d).items() if k in ("read_order", "max_modulus", "max_gap", "y_last")}
                                    for rule in mod["RULES"] for d in (0, 1)}
res["sweep"] = sweep

# 5) 연습문제 답
m10 = load(T=0.01)
ex1 = {f"delay{d}": {k: v for k, v in case(m10, "ZOH, Forward Euler", d).items() if k in ("read_order", "read_poles", "max_modulus")} for d in (0, 1)}
c2 = case(load(T=0.02), "ZOH, Forward Euler", 2)
ex2 = {k: c2[k] for k in ("read_order", "read_poles", "max_modulus", "max_gap")}
m40 = load(T=0.04)
ex3 = {f"{rule}|delay{d}": round(case(m40, rule, d)["max_modulus"], 4) for rule in m40["RULES"] for d in (0, 1)}
res["exercises"] = {"1_T10ms_ZOH_FE": ex1, "2_delay2_T20ms_ZOH_FE": ex2, "3_T40ms_max_modulus": ex3}

json.dump(res, open("results_f4d3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved results_f4d3.json")

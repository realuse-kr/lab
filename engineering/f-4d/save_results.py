"""F-4D 검증 계산: 결과를 results_f4d.json에 저장한다."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from plant_zoh import TAU, METHODS, discretize, step_error

out = {"tau_s": TAU, "env": "Python 3.13, NumPy 2.5.3, SciPy 1.18.1"}

# 1) T = 20 ms: 극점, 계수, 최대 응답 오차, k=0 샘플 값
T = 0.02
res = {}
for name, m in METHODS.items():
    num, den = discretize(m, T)
    e = step_error(m, T)
    i = int(np.argmax(np.abs(e)))
    res[name] = {"pole": float(-den[1]), "b0": float(num[0]), "b1": float(num[1]),
                 "max_err_pct": float(e[i]), "max_err_at_ms": i * T * 1e3,
                 "err_k0_pct": float(e[0]), "err_k1_pct": float(e[1])}
out["T20ms"] = res
out["exp_minus_T_over_tau"] = float(np.exp(-T / TAU))

# 2) 주기별 극점과 최대 응답 오차
sweep = []
for Tm in (0.005, 0.01, 0.02, 0.04, 0.1):
    row = {"T_ms": Tm * 1e3}
    for name, m in METHODS.items():
        num, den = discretize(m, Tm)
        e = step_error(m, Tm)
        row[name] = {"pole": float(-den[1]), "max_err_pct": float(e[np.argmax(np.abs(e))])}
    sweep.append(row)
out["sweep"] = sweep

# 3) 매 샘플 일정하게 유지되는 임의 입력: ZOH 모델과 연속 시간 ODE(rtol 1e-12) 비교
rng = np.random.default_rng(0)
u = rng.uniform(-1, 1, 100)
num, den = discretize("zoh", T)
yd = lfilter(num, den, u)
x, ys = 0.0, []
for uk in u:
    ys.append(x)
    sol = solve_ivp(lambda t, y: (-y + uk) / TAU, (0, T), [x], rtol=1e-12, atol=1e-14)
    x = sol.y[0, -1]
yc = np.array(ys)  # y[k] = x(kT); 샘플 시점의 연속 시간 해
# ZOH 모델 y[k+1] = a y[k] + (1-a) u[k]를 직접 재귀로 계산해 연속 시간 해, lfilter 결과와 비교한다
a = np.exp(-T / TAU)
yr = [0.0]
for uk in u[:-1]:
    yr.append(a * yr[-1] + (1 - a) * uk)
out["random_held_input_max_abs_diff"] = float(np.max(np.abs(np.array(yr) - yc)))
out["lfilter_vs_recursion_max_abs_diff"] = float(np.max(np.abs(np.array(yr) - yd)))

# 4) Tustin의 k=0 값 = b0 = T/(2 TAU + T), Backward Euler b0 = T/(TAU + T)
out["tustin_b0_formula"] = float(T / (2 * TAU + T))
out["backward_b0_formula"] = float(T / (TAU + T))

# 5) 연습문제 답
ex = {}
import plant_zoh
plant_zoh.TAU = 0.05
n, d = plant_zoh.discretize("zoh", 0.02)
ex["ex1_tau0.05_zoh_pole_T20ms"] = float(-d[1])
plant_zoh.TAU = 0.1
n, d = plant_zoh.discretize("euler", 0.1)
ex["ex2_forward_pole_T100ms"] = float(-d[1])
n, d = plant_zoh.discretize("backward_diff", 0.01)
ex["ex3_backward_b0_T10ms_pct"] = float(n[0] * 100)
out["exercises"] = ex

with open("results_f4d.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

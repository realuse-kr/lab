import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, lfilter, step, TransferFunction
import controller_rules as cr

PERIODS_MS = [5, 10, 20, 40]

def peak(e, T):  # 절댓값이 가장 큰 편차(부호 유지)와 그 시각 [ms]
    i = int(np.argmax(np.abs(e)))
    return float(e[i]), float(round(i * T * 1e3, 6))

def plant_step_error(rule, T, t_end=2.0):  # 플랜트 이산화만 본 스텝 응답 오차 [% of 최종값 1]
    num, den, _ = cont2discrete(([1.0], [cr.TAU, 1.0]), T, method=rule)
    k = np.arange(int(round(t_end / T)) + 1)
    y = lfilter(np.squeeze(num), np.squeeze(den), np.ones(k.size))
    return (y - (1 - np.exp(-k * T / cr.TAU))) * 100

def continuous_ref(T, t_end=1.0):  # 연속 폐루프 스텝 응답(독립 작성: 폐루프 전달함수 + scipy step)
    sys = TransferFunction([cr.KP, cr.KI], [cr.TAU, 1 + cr.KP, cr.KI])
    t = np.arange(int(round(t_end / T)) + 1) * T
    return step(sys, T=t)[1]

def hybrid_loop(rule, T, t_end=1.0):  # 연속 플랜트를 ODE로 적분하고 입력은 샘플 사이에 일정하게 유지
    c0, c1, d1 = cr.controller(rule, T)
    n = int(round(t_end / T)) + 1
    y = np.zeros(n)
    x = e_old = u_old = 0.0
    for k in range(n):
        y[k] = x
        e = 1.0 - x
        u = c0 * e + c1 * e_old - d1 * u_old
        x = solve_ivp(lambda t, s: [(-s[0] + u) / cr.TAU], (0, T), [x], method="DOP853", rtol=1e-12, atol=1e-14).y[0, -1]
        e_old, u_old = e, u
    return y

def coeff_formula(rule, T):  # 손으로 유도한 PI 차분방정식 계수 (c0, c1)
    kp, ki = cr.KP, cr.KI
    return {"ZOH": (kp, -kp + ki * T), "Forward Euler": (kp, -kp + ki * T),
            "Backward Euler": (kp + ki * T, -kp), "Tustin": (kp + ki * T / 2, -kp + ki * T / 2)}[rule]

res = {"params": {"TAU": cr.TAU, "KP": cr.KP, "KI": cr.KI, "periods_ms": PERIODS_MS, "t_end_s": 1.0}}
T = 0.02
res["coefficients_T20ms"] = {n: dict(zip(("c0", "c1", "d1"), (round(float(v), 10) for v in cr.controller(r, T)))) for n, r in cr.RULES.items()}
res["zoh_vs_forward_euler_coeff_max_abs_diff"] = float(max(abs(a - b) for a, b in zip(cr.controller("zoh", T), cr.controller("euler", T))))
res["formula_vs_scipy_coeff_max_abs_diff"] = float(max(np.abs(np.array(cr.controller(r, T)[:2]) - np.array(coeff_formula(n, T))).max() for n, r in cr.RULES.items()))
res["sweep"] = {f"{ms}ms": {n: dict(zip(("max_dev_pct", "at_ms"), peak(cr.deviation_pct(r, ms / 1e3), ms / 1e3))) for n, r in cr.RULES.items()} for ms in PERIODS_MS}
res["first_sample_dev_pct_T20ms"] = {n: float(cr.deviation_pct(r, T)[1]) for n, r in cr.RULES.items()}
res["plant_side_T20ms"] = {n: {"max_err_pct": peak(plant_step_error(r, T), T)[0]} for n, r in cr.RULES.items()}
res["plant_exp_minus_T_over_tau"] = float(np.exp(-T / cr.TAU))

yc = continuous_ref(T)
res["continuous_loop_T20ms"] = {"error_at_0_ms": 1.0, "error_at_20_ms": float(1.0 - yc[1]), "y_max": float(yc.max())}
# 독립 검증 두 가지(단위: 최종값 1 대비 비율): (1) 이산 루프(ZOH 플랜트 모델) = 연속 플랜트 ODE에 일정 입력을 넣은 루프, (2) ODE 기준 응답 = 폐루프 전달함수 스텝 응답
y_ode = solve_ivp(lambda _, s: [(-s[0] + cr.KP * (1 - s[0]) + cr.KI * s[1]) / cr.TAU, 1 - s[0]], (0, 1), [0, 0],
                  t_eval=np.arange(51) * T, method="DOP853", rtol=1e-12, atol=1e-14).y[0]
res["loop_vs_hybrid_ode_max_abs_diff"] = float(max(np.abs(cr.deviation_pct(r, T) / 100 + y_ode - hybrid_loop(r, T)).max() for r in cr.RULES.values()))
res["continuous_ref_vs_ode_max_abs_diff"] = float(np.abs(y_ode - yc).max())

ex = {"1_T10ms": res["sweep"]["10ms"]}
saved = cr.KI
cr.KI = 20.0
ex["2_KI20_T20ms"] = {n: {"c1": round(float(cr.controller(r, T)[1]), 10), **dict(zip(("max_dev_pct", "at_ms"), peak(cr.deviation_pct(r, T), T)))} for n, r in cr.RULES.items()}
cr.KI = saved
ex["3_formula"] = {"ZOH_and_Forward_Euler": "c0=KP, c1=-KP+KI*T", "Backward Euler": "c0=KP+KI*T, c1=-KP",
                   "Tustin": "c0=KP+KI*T/2, c1=-KP+KI*T/2", "d1": -1.0}
res["exercise_answers"] = ex

with open("results_f4d2.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "exercise_answers"}, ensure_ascii=False, indent=1))

"""F-5.2 결과를 results_f52.json으로 저장한다.
두 표기의 응답을 전달함수 경로와 시간 영역 적분 경로로 각각 구해 최대 차이를 잰다."""
import json
import numpy as np
from scipy import signal
import pid_form_equivalence as m

out = {"plant": "1/((0.1 s+1)(0.02 s+1))", "target": 1.0}

# 1. 기본 숫자 한 벌: 이상형 Kc 2, Ti 0.1, Td 0.02 <-> 병렬형 Kp 2, Ki 20, Kd 0.04
kc, ti, td = 2.0, 0.1, 0.02
kp, ki, kd = m.ideal_to_parallel(kc, ti, td)
t = np.linspace(0, 1.0, 100001)
y_par, den_par = m.step_tf_parallel(kp, ki, kd, t)
y_ide, den_ide = m.step_tf_ideal(kc, ti, td, t)
y_par_ode = m.step_ode_parallel(kp, ki, kd, t)
y_ide_ode = m.step_ode_ideal(kc, ti, td, t)
y_wrong, _ = m.step_tf_parallel(kc, ti, td, t)  # 이상형 숫자를 병렬형에 그대로

def diff(a, b):
    return float(np.abs(a - b).max())

out["main"] = {
    "ideal": {"Kc": kc, "Ti": ti, "Td": td},
    "parallel": {"Kp": kp, "Ki": ki, "Kd": kd},
}
for name, y in (("parallel", y_par), ("ideal", y_ide)):
    os_, ts = m.metrics(t, y)
    out["main"][name].update({
        "y_at_0.6s": round(float(y[60000]), 4),
        "overshoot_pct": round(float(os_), 2),
        "settling_2pct_s": round(float(ts), 4),
    })
out["max_diff"] = {
    "tf_parallel_vs_tf_ideal": diff(y_par, y_ide),
    "ode_parallel_vs_ode_ideal": diff(y_par_ode, y_ide_ode),
    "tf_vs_ode_parallel": diff(y_par, y_par_ode),
    "tf_vs_ode_ideal": diff(y_ide, y_ide_ode),
    "ideal_numbers_into_parallel_vs_converted": diff(y_wrong, y_par),
}

# 2. 폐루프 극점: 두 표기의 분모 다항식을 각각 monic으로 만들어 근을 비교한다
p_par = np.sort_complex(np.roots(den_par))
p_ide = np.sort_complex(np.roots(den_ide))
out["poles"] = {
    "parallel": [round(float(v.real), 4) if abs(v.imag) < 1e-9 else [round(float(v.real), 4), round(float(v.imag), 4)] for v in p_par],
    "max_abs_diff": float(np.abs(p_par - p_ide).max()),
    "stable": bool(np.all(p_par.real < 0) and np.all(p_ide.real < 0)),
}

# 3. scipy.signal.step 대조 (0~0.6 s)
t6 = np.linspace(0, 0.6, 60001)
num_par = np.array([kd, kp, ki])
y_sci = signal.step(signal.TransferFunction(num_par, den_par), T=t6)[1]
out["max_diff_vs_scipy_step"] = diff(m.step_tf_parallel(kp, ki, kd, t6)[0], y_sci)

# 4. 무작위 게인 묶음: 두 방향으로 변환한 뒤 두 표기의 응답을 비교한다 (전달함수 경로, 0~1 s 2001점)
rng = np.random.default_rng(52)
ts = np.linspace(0, 1.0, 2001)
N = 1000

def loguni(lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

def sweep(sets, to_other, direction):
    n_stable, worst, worst_rt = 0, 0.0, 0.0
    for a in sets:
        if direction == "ideal_to_parallel":
            kc_, ti_, td_ = a
            kp_, ki_, kd_ = to_other(*a)
            round_trip = m.parallel_to_ideal(kp_, ki_, kd_)
        else:
            kp_, ki_, kd_ = a
            kc_, ti_, td_ = to_other(*a)
            round_trip = m.ideal_to_parallel(kc_, ti_, td_)
        ya, da = m.step_tf_parallel(kp_, ki_, kd_, ts)
        yb, db = m.step_tf_ideal(kc_, ti_, td_, ts)
        if not (np.all(np.roots(da).real < 0) and np.all(np.roots(db).real < 0)):
            continue
        n_stable += 1
        worst = max(worst, diff(ya, yb))
        worst_rt = max(worst_rt, max(abs(r - o) / abs(o) for r, o in zip(round_trip, a)))
    return {"sets": len(sets), "stable_sets": n_stable,
            "max_step_diff": worst, "max_round_trip_rel_err": worst_rt}

sets_a = list(zip(loguni(0.5, 20, N), loguni(0.05, 2, N), loguni(0.005, 0.1, N)))  # Kc, Ti, Td
sets_b = list(zip(loguni(0.5, 20, N), loguni(1, 500, N), loguni(0.005, 1, N)))      # Kp, Ki, Kd
out["sweep"] = {
    "ideal_to_parallel": sweep(sets_a, m.ideal_to_parallel, "ideal_to_parallel"),
    "parallel_to_ideal": sweep(sets_b, m.parallel_to_ideal, "parallel_to_ideal"),
}
out["sweep_ranges"] = {"rng_seed": 52, "sampling": "log-uniform", "sets_per_direction": N,
    "ideal": {"Kc": [0.5, 20], "Ti": [0.05, 2], "Td": [0.005, 0.1]},
    "parallel": {"Kp": [0.5, 20], "Ki": [1, 500], "Kd": [0.005, 1]}}

# 5. 시간 영역 적분 경로로 다시 확인: 안정한 묶음 앞쪽 100벌씩 (0~1 s 2001점)
def ode_check(sets, to_other, direction, n=100):
    worst, cnt = 0.0, 0
    for a in sets:
        if cnt >= n:
            break
        if direction == "ideal_to_parallel":
            kc_, ti_, td_ = a
            kp_, ki_, kd_ = to_other(*a)
        else:
            kp_, ki_, kd_ = a
            kc_, ti_, td_ = to_other(*a)
        _, da = m.step_tf_parallel(kp_, ki_, kd_, ts)
        if not np.all(np.roots(da).real < 0):
            continue
        cnt += 1
        worst = max(worst, diff(m.step_ode_parallel(kp_, ki_, kd_, ts), m.step_ode_ideal(kc_, ti_, td_, ts)))
    return {"sets": cnt, "max_step_diff": worst}

out["sweep_ode"] = {
    "ideal_to_parallel": ode_check(sets_a, m.ideal_to_parallel, "ideal_to_parallel"),
    "parallel_to_ideal": ode_check(sets_b, m.parallel_to_ideal, "parallel_to_ideal"),
}

with open("results_f52.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
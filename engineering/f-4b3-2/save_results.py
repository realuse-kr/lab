"""F-4B3.2: sampling_sweep.py의 결과를 results_f4b3_2.json으로 저장하고, 계수·극점·분모를 다른 경로로 다시 구해 대조한다."""
import json
import numpy as np
from scipy.signal import cont2discrete
from sampling_sweep import (KP, KI, TAU, DIRECT, EMULATION, T_MS_MAX, TOLS, target_poles, direct_design,
                            coefficients, closed_loop_poles, pole_abs_max, closed_loop_deviation,
                            longest_period, first_unstable)

RULES = EMULATION + [DIRECT]
GRID_MS = list(range(1, T_MS_MAX + 1))
TABLE_MS = [5, 10, 20, 40, 80]

# F-4B3의 results_f4b3.json에 저장된 T = 20 ms의 폐루프 편차. 같은 식이므로 이 값과 같아야 한다.
F4B3_DEVIATION_20MS = {
    "Forward Euler": 0.1180136056812473, "Backward Euler": 0.17955113799890476, "Tustin": 0.12411695856808957,
    "스텝 불변": 0.1180136056812473, "극·영점 매칭": 0.12636152462873496, "프리워핑 Tustin": 0.1303322752161945,
    DIRECT: 0.04559244617380942,
}

grid = {r: {"deviation": [], "pole_abs_max": []} for r in RULES}
for t_ms in GRID_MS:
    T = t_ms / 1000
    for r in RULES:
        c = coefficients(r, T)
        grid[r]["deviation"].append(closed_loop_deviation(c, T))
        grid[r]["pole_abs_max"].append(pole_abs_max(c, T))

rows = {}
for t_ms in TABLE_MS:
    T = t_ms / 1000
    rows[str(t_ms)] = {}
    for r in RULES:
        c = coefficients(r, T)
        zmax = pole_abs_max(c, T)
        rows[str(t_ms)][r] = {"c0": float(c[0]), "c1": float(c[1]), "deviation": closed_loop_deviation(c, T),
                              "pole_abs_max": zmax, "stable": bool(zmax < 1.0)}
    emu = [rows[str(t_ms)][r]["deviation"] for r in EMULATION]
    rows[str(t_ms)]["best_emulation_rule"] = EMULATION[int(np.argmin(emu))]
    rows[str(t_ms)]["best_emulation_deviation"] = float(min(emu))
    rows[str(t_ms)]["ratio_best_emulation_to_direct"] = float(min(emu) / rows[str(t_ms)][DIRECT]["deviation"])

out = {
    "plant_tau_s": TAU, "kp": KP, "ki": KI,
    "continuous_closed_loop_poles_rad_per_s": [[float(p.real), float(p.imag)] for p in np.roots([TAU, 1 + KP, KI])],
    "grid_ms": GRID_MS,
    "rows": rows,
    "longest_period_ms": {f"{tol:.0%}": {r: longest_period(r, tol) for r in RULES} for tol in TOLS},
    "first_unstable_ms": {r: first_unstable(r) for r in RULES},
    "grid": grid,
}

# 점검 1: 직접 이산 설계의 계수를 선형 방정식 풀이로 다시 구한다.
ok1 = True
for t_ms in GRID_MS:
    T = t_ms / 1000
    pn, pd, _ = cont2discrete(([1.0], [TAU, 1.0]), T, "zoh")[:3]
    b, a = float(np.ravel(pn)[-1]), float(np.ravel(pd)[-1])  # P(z) = b/(z + a)
    tp = np.poly(target_poles(T)).real  # z^2 + a1 z + a0
    # (z - 1)(z + a) + b (c0 z + c1) = z^2 + (a - 1 + b c0) z + (-a + b c1)
    c_lin = np.linalg.solve(np.array([[b, 0.0], [0.0, b]]), np.array([tp[1] - (a - 1.0), tp[2] + a]))
    ok1 = ok1 and bool(np.allclose(c_lin, direct_design(T), atol=1e-10))
out["check_direct_coefficients_match"] = ok1


# 점검 2: 폐루프를 상태 행렬로 쓴 고유값이 특성다항식의 근과 같다(고유값 하나는 0).
def eigen(c, T):
    al = float(np.exp(-T / TAU))
    m3 = np.array([[al - (1 - al) * c[0], (1 - al), (1 - al) * c[1]], [-c[0], 1.0, c[1]], [-1.0, 0.0, 0.0]])
    return sorted([e for e in np.linalg.eigvals(m3) if abs(e) > 1e-9], key=lambda x: (x.real, x.imag))


ok2 = True
for t_ms in GRID_MS:
    T = t_ms / 1000
    for r in RULES:
        c = coefficients(r, T)
        roots = sorted(closed_loop_poles(c, T), key=lambda x: (x.real, x.imag))
        ev = eigen(c, T)
        ok2 = ok2 and len(ev) == 2 and bool(np.allclose(roots, ev, rtol=1e-7, atol=1e-9))
out["check_eigenvalues_match"] = ok2

# 점검 3: 직접 이산 설계의 폐루프 극점이 모든 T에서 목표 극점과 같고, 목표 극점 크기가 exp(-15 T)와 같다.
ok3, ok4 = True, True
for t_ms in GRID_MS:
    T = t_ms / 1000
    r = sorted(closed_loop_poles(direct_design(T), T), key=lambda x: (x.real, x.imag))
    zt = sorted(target_poles(T), key=lambda x: (x.real, x.imag))
    ok3 = ok3 and bool(np.allclose(r, zt, atol=1e-9))
    ok4 = ok4 and bool(np.allclose(abs(zt[0]), np.exp(-15.0 * T), atol=1e-12))
out["check_direct_poles_equal_target"] = ok3
out["check_target_abs_is_exp_minus_15T"] = ok4

# 점검 4: 연속 시간 폐루프를 스텝 입력에 ZOH로 이산화한 기준 시스템의 분모가 목표 다항식과 같다.
ok5 = True
for t_ms in GRID_MS:
    T = t_ms / 1000
    _, cd, _ = cont2discrete(([KP, KI], [TAU, 1 + KP, KI]), T, "zoh")[:3]
    ok5 = ok5 and bool(np.allclose(np.ravel(cd), np.poly(target_poles(T)).real, atol=1e-9))
out["check_reference_denominator_match"] = ok5

# 점검 5: T = 20 ms의 폐루프 편차가 F-4B3의 저장값과 같다.
out["check_matches_f4b3_at_20ms"] = bool(all(
    abs(rows["20"][r]["deviation"] - v) < 1e-12 for r, v in F4B3_DEVIATION_20MS.items()))

# 점검 6: 표의 샘플링 주기에서 (에뮬레이션 중 가장 작은 편차)/(직접 이산 설계 편차)가 샘플링 주기와 함께 커진다.
ratios = [rows[str(t)]["ratio_best_emulation_to_direct"] for t in TABLE_MS]
out["check_ratio_increases_across_table"] = bool(all(b > a for a, b in zip(ratios, ratios[1:])))

with open("results_f4b3_2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k.startswith("check_")})
print("ratios", [round(x, 2) for x in ratios])

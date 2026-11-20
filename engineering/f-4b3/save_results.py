"""F-4B3: direct_design.py의 결과를 results_f4b3.json으로 저장하고, 손으로 푼 계수와 폐루프 극점을 다른 경로로 점검한다."""
import json
import numpy as np
from scipy.signal import cont2discrete
from direct_design import (KP, KI, TAU, T_DEFAULT, target_poles, direct_design, emulation_rules,
                           closed_loop_poles, pole_distance, closed_loop_deviation)

T = T_DEFAULT
alpha = float(np.exp(-T / TAU))
zt = target_poles(T)
c_direct = direct_design(T)
rows = dict(emulation_rules(T))
rows["직접 이산 설계"] = c_direct

out = {
    "T_s": T,
    "plant": {"tau_s": TAU, "alpha": alpha},
    "continuous_closed_loop_poles_rad_per_s": [[float(p.real), float(p.imag)] for p in np.roots([TAU, 1 + KP, KI])],
    "target_poles_z": [[float(p.real), float(p.imag)] for p in zt],
    "target_abs_z": float(abs(zt[0])),
    "rows": {},
}
for name, c in rows.items():
    r = sorted(closed_loop_poles(c, T), key=lambda x: x.imag)[1]
    out["rows"][name] = {
        "c0": float(c[0]), "c1": float(c[1]),
        "zero_z": float(-c[1] / c[0]),
        "low_freq_integral_gain": float((c[0] + c[1]) / T),
        "closed_loop_pole_z": [float(r.real), float(r.imag)],
        "closed_loop_pole_abs": float(abs(r)),
        "distance_to_target": pole_distance(c, T),
        "closed_loop_deviation": closed_loop_deviation(c, T),
    }

# 점검 1: 계수를 손으로 푼 식과 다른 경로(선형 방정식 풀이)가 같은 값을 주는가
pn, pd, _ = cont2discrete(([1.0], [TAU, 1.0]), T, "zoh")[:3]
b = float(np.ravel(pn)[-1])
a = float(np.ravel(pd)[-1])  # P(z) = b/(z + a), a = -alpha
target_poly = np.poly(zt).real  # z^2 + a1 z + a0
# (z - 1)(z + a) + b (c0 z + c1) = z^2 + (a - 1 + b c0) z + (-a + b c1)
M = np.array([[b, 0.0], [0.0, b]])
rhs = np.array([target_poly[1] - (a - 1.0), target_poly[2] + a])
c_lin = np.linalg.solve(M, rhs)
out["check_coefficients_match"] = bool(np.allclose(c_lin, c_direct, atol=1e-12))

# 점검 2: 폐루프를 상태 행렬로 쓴 고유값이 특성다항식의 근과 같은가(고유값 하나는 0)
def eigen(c):
    al = alpha
    M3 = np.array([[al - (1 - al) * c[0], (1 - al), (1 - al) * c[1]], [-c[0], 1.0, c[1]], [-1.0, 0.0, 0.0]])
    ev = [e for e in np.linalg.eigvals(M3) if abs(e) > 1e-9]
    return sorted(ev, key=lambda x: x.imag)
ok = True
for name, c in rows.items():
    r = sorted(closed_loop_poles(c, T), key=lambda x: x.imag)
    ev = eigen(c)
    ok = ok and len(ev) == 2 and all(abs(x - y) < 1e-9 for x, y in zip(r, ev))
out["check_eigenvalues_match"] = bool(ok)

# 편차가 0이 아닌 이유: 극점은 같고 분자(영점)가 다르다
cn, cd, _ = cont2discrete(([KP, KI], [TAU, 1 + KP, KI]), T, "zoh")[:3]
cn = np.trim_zeros(np.ravel(cn), "f")
out["zoh_equivalent_closed_loop"] = {
    "numerator": [float(x) for x in cn], "denominator": [float(x) for x in np.ravel(cd)],
    "zero_z": float(np.roots(cn)[0].real)}
out["direct_closed_loop"] = {
    "numerator": [float((1 - alpha) * c_direct[0]), float((1 - alpha) * c_direct[1])],
    "denominator": [float(x) for x in target_poly],
    "zero_z": float(-c_direct[1] / c_direct[0])}
out["denominators_match"] = bool(np.allclose(np.ravel(cd), target_poly, atol=1e-12))

with open("results_f4b3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("check_coefficients_match", out["check_coefficients_match"],
      "check_eigenvalues_match", out["check_eigenvalues_match"], "denominators_match", out["denominators_match"])
print("zero ZOH-equivalent", out["zoh_equivalent_closed_loop"]["zero_z"], "direct", out["direct_closed_loop"]["zero_z"])

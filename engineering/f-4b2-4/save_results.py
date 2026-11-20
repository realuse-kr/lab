"""F-4B2.4: 여섯 규칙 비교 결과를 results_f4b24.json으로 저장하고, 네 규칙의 계수를 SciPy cont2discrete와 독립 대조한다."""
import json
import numpy as np
from scipy.signal import cont2discrete
from six_rules import (KP, KI, crossover_and_bandwidth, rules, ratio, controller_out,
                       closed_loop_deviation)

RATIOS = (6, 8, 10, 20, 25, 30, 40, 100)
NAMES = ["Forward Euler", "Backward Euler", "Tustin", "스텝 불변", "극·영점 매칭", "프리워핑 Tustin"]
wc, fb = crossover_and_bandwidth()
T = 0.02
t = np.arange(101) * T
out = {"wc_rad_s": wc, "fb_hz": fb, "m_at_T20": 1 / (T * fb), "T20": {}, "sweep": {}}
for n, c in rules(T).items():
    r = ratio(c, T, wc)
    step = controller_out(c, np.ones(t.size)) - (KP + KI * t)
    ramp = controller_out(c, t) - (KP * t + KI * t ** 2 / 2)
    out["T20"][n] = {"c0": c[0], "c1": c[1], "zero": -c[1] / c[0], "low_freq_integral_gain": (c[0] + c[1]) / T,
                     "mag_ratio_wc": abs(r), "phase_diff_deg_wc": float(np.degrees(np.angle(r))),
                     "step_diff_2s": float(step[-1]), "step_diff_max_abs": float(np.max(np.abs(step))),
                     "ramp_diff_2s": float(ramp[-1]), "closed_loop_deviation": closed_loop_deviation(c, T)}
for m in RATIOS:
    Tm = 1 / (m * fb)
    out["sweep"][str(m)] = {"T_ms": Tm * 1e3, **{n: closed_loop_deviation(c, Tm) for n, c in rules(Tm).items()}}

# 독립 대조: SciPy cont2discrete로 C(s) = (KP s + KI)/s를 이산화한 계수와 rules()의 계수를 비교한다.
# 'zoh'는 스텝 불변, 'euler'는 Forward Euler, 'backward_diff'는 Backward Euler, 'bilinear'는 Tustin이다.
check = {}
for name, method in (("스텝 불변", "zoh"), ("Forward Euler", "euler"), ("Backward Euler", "backward_diff"), ("Tustin", "bilinear")):
    num, den, _ = cont2discrete(([KP, KI], [1.0, 0.0]), T, method=method)
    assert np.allclose(den, [1.0, -1.0])
    check[name] = bool(np.allclose(np.ravel(num), rules(T)[name], atol=1e-12))
out["scipy_cont2discrete_match"] = check
json.dump(out, open("results_f4b24.json", "w"), ensure_ascii=False, indent=1)
print("SciPy cont2discrete 대조:", check)
print("스텝 입력 차이 최대 절댓값(스텝 불변):", out["T20"]["스텝 불변"]["step_diff_max_abs"])
for m in RATIOS:
    d = out["sweep"][str(m)]
    print(m, [f"{n}={d[n] * 100:.2f}" for n in sorted(NAMES, key=lambda n: d[n])])

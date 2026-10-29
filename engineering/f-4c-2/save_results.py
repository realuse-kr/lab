"""F-4C.2 검증 계산: 본문 숫자와 그림 데이터를 results_f4c2.json에 저장한다."""
import json, platform
import numpy as np, scipy
from scipy.signal import cont2discrete, freqz
from scipy.optimize import brentq
from pi_freq_response import KP, KI, METHODS, mag_error

TAU = 0.1  # 플랜트 시간상수 [s] (이득 교차 주파수 계산에만 사용)
Cc = lambda w: KP + KI / (1j * w)

def coef(m, T):
    num, den, _ = cont2discrete(([KP, KI], [1, 0]), T, method=m)
    return np.squeeze(num), np.squeeze(den)

def Cd(m, T, w):
    b, a = coef(m, T); z = np.exp(1j * w * T)
    return np.polyval(b, z) / np.polyval(a, z)

def first_over(e, w, lim=1.0):
    o = np.abs(e) > lim
    return float(w[np.argmax(o)]) if o.any() else None

res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
       "params": {"Kp": KP, "Ki": KI, "tau_plant_s": TAU, "grid": "geomspace(0.1, pi/T, 400001)",
                  "mag_error_def": "(|C(e^{jwT})|/|C(jw)| - 1)*100 [%]", "threshold_pct": 1.0}}

# 1) T = 20 ms 상세
T = 0.02; w = np.geomspace(0.1, np.pi / T, 400001)
d = {"nyquist_rad_s": np.pi / T, "nyquist_hz": 1 / (2 * T)}
for n, m in METHODS.items():
    e, b = mag_error(m, T, w); i = np.argmax(np.abs(e))
    ph = np.degrees(np.angle(Cd(m, T, w)) - np.angle(Cc(w)))
    d[n] = {"b0": float(b[0]), "b1": float(b[1]), "max_abs_mag_err_pct": float(e[i]),
            "at_rad_s": float(w[i]), "mag_err_min_pct": float(e.min()), "mag_err_max_pct": float(e.max()),
            "first_over_1pct_rad_s": first_over(e, w), "mag_err_at_nyquist_pct": float(e[-1]),
            "phase_err_min_deg": float(ph.min()), "phase_err_max_deg": float(ph.max())}
res["T20ms"] = d
# 비례항과 적분항의 크기가 같아지는 Ki/Kp [rad/s]에서의 크기 오차
wk = np.array([KI / KP])
res["T20ms"]["at_Ki_over_Kp"] = {"w_rad_s": KI / KP, **{n: float(mag_error(m, T, wk)[0][0]) for n, m in METHODS.items()}}

# 2) 등가 관계: Forward/Backward Euler = Tustin에서 Kp를 Ki*T/2만큼 빼거나 더한 것
off = KI * T / 2
def tustin_with_kp(kp):
    num, _, _ = cont2discrete(([kp, KI], [1, 0]), T, method="bilinear")
    return np.squeeze(num)
res["identity_T20ms"] = {
    "Ki_T_over_2": off,
    "FE_equals_Tustin_Kp_minus": bool(np.allclose(coef("euler", T)[0], tustin_with_kp(KP - off))),
    "BE_equals_Tustin_Kp_plus": bool(np.allclose(coef("backward_diff", T)[0], tustin_with_kp(KP + off)))}
# Forward Euler에 Kp 대신 Kp + Ki*T/2를 넣으면 Tustin과 계수가 같은지
num_fe, _, _ = cont2discrete(([KP + off, KI], [1, 0]), T, method="euler")
res["identity_T20ms"]["FE_with_Kp_plus_KiT2_equals_Tustin"] = bool(np.allclose(np.squeeze(num_fe), coef("bilinear", T)[0]))
# 반대 부호 검사(변이): Kp - Ki*T/2를 넣으면 같지 않아야 한다
num_bad, _, _ = cont2discrete(([KP - off, KI], [1, 0]), T, method="euler")
res["identity_T20ms"]["mutation_wrong_sign_equals_Tustin"] = bool(np.allclose(np.squeeze(num_bad), coef("bilinear", T)[0]))

# 3) Tustin 적분항만의 크기 오차와 PI 전체 크기 오차 비교
xi = np.array([0.25, 0.5, 0.75, 0.999]) * np.pi / T  # 나이퀴스트의 25/50/75/99.9%
integ = (xi * T / 2) / np.tan(xi * T / 2)  # |Tustin 적분항| / |1/(jw)|
res["tustin_integral_term_T20ms"] = [{"w_rad_s": float(x), "integral_mag_err_pct": float((r - 1) * 100),
    "pi_mag_err_pct": float(mag_error("bilinear", T, np.array([x]))[0][0])} for x, r in zip(xi, integ)]

# 4) 이득 교차 주파수(연속 시간 개루프 |C P| = 1)에서의 오차
wc = brentq(lambda x: abs(Cc(x) / (TAU * 1j * x + 1)) - 1, 1, 1000)
res["crossover"] = {"wc_rad_s": wc}
for n, m in METHODS.items():
    v = Cd(m, T, np.array([wc]))[0]
    res["crossover"][n] = {"mag_err_pct": float((abs(v) / abs(Cc(wc)) - 1) * 100),
                           "phase_err_deg": float(np.degrees(np.angle(v) - np.angle(Cc(wc))))}

# 5) 주기별 1% 초과 주파수
res["sweep"] = []
for Ts in (0.005, 0.01, 0.02, 0.04):
    ww = np.geomspace(0.1, np.pi / Ts, 400001); row = {"T_ms": Ts * 1e3, "nyquist_rad_s": np.pi / Ts}
    for n, m in METHODS.items():
        e, _ = mag_error(m, Ts, ww)
        row[n] = {"first_over_1pct_rad_s": first_over(e, ww), "max_abs_mag_err_pct": float(e[np.argmax(np.abs(e))])}
    res["sweep"].append(row)

# 6) Ki를 80으로 바꾼 경우(T = 20 ms, 연습문제 3)
import pi_freq_response as pf
pf.KI = 80.0
res["Ki80_T20ms"] = {}
for n, m in METHODS.items():
    e, b = pf.mag_error(m, T, w)
    res["Ki80_T20ms"][n] = {"b0": float(b[0]), "b1": float(b[1]), "max_abs_mag_err_pct": float(e[np.argmax(np.abs(e))]),
                            "first_over_1pct_rad_s": first_over(e, w)}
pf.KI = KI

# 7) 검산: SciPy freqz와 직접 계산 비교, 격자 10배 조밀
fz = 0.0
for n, m in METHODS.items():
    b, a = coef(m, T); _, h = freqz(b, a, worN=w[::100] * T)
    fz = max(fz, float(np.max(np.abs(h - Cd(m, T, w[::100])))))
res["check"] = {"freqz_max_abs_diff": fz}
w10 = np.geomspace(0.1, np.pi / T, 4000001)
res["check"]["first_over_1pct_fine_grid"] = {n: first_over(mag_error(m, T, w10)[0], w10) for n, m in METHODS.items()}

with open("results_f4c2.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))

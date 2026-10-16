"""F-2 극점으로 2차계 스텝 응답 읽기. 가상 파라미터 시뮬레이션.

계산 내용
  A. 극점 위치와 식(오버슈트, 피크 시간)을 수치 응답과 대조
  B. 정정시간 근사 4/(ζωn)의 오차를 ζ 스윕으로 확인
  C. 같은 극점에 영점을 더했을 때(왼쪽·오른쪽 반평면) 응답 변화, ỹ = y + ẏ/α 항등식 확인
  D. 같은 폐루프의 입력 경로(기준 입력 T, 출력 외란 S)에 따라 달라지는 응답
  E. 초기조건 응답과 스텝 응답의 관계(x0=1의 자유응답 = 1 - 스텝 응답)
결과는 results/F-2/results.json에 쓴다.
"""
import json
import numpy as np
from scipy import signal

from bandwidth import ROOT, step_metrics


def std_tf(zeta, wn, num=None):
    den = [1.0, 2 * zeta * wn, wn ** 2]
    return signal.TransferFunction(num if num is not None else [wn ** 2], den)


def zero_tf(zeta, wn, alpha):
    """극점은 그대로, 영점 s=-alpha 추가. DC 이득 1로 정규화: (s/alpha + 1) * wn^2 / (...).
    alpha < 0이면 오른쪽 반평면 영점(1 - s/|alpha|)."""
    return std_tf(zeta, wn, [wn ** 2 / alpha, wn ** 2])


def step_y(sys, t):
    _, y = signal.step(sys, T=t)
    return y


def extra_metrics(sys, t_end, dt):
    t = np.arange(0.0, t_end, dt)
    y = step_y(sys, t)
    ip = int(np.argmax(y))
    m = step_metrics(sys, t_end, dt)
    m["peak_time_s"] = float(t[ip])
    m["undershoot_pct"] = float(max(0.0, -100.0 * y.min()))
    m["t_undershoot_s"] = float(t[int(np.argmin(y))])
    return m


def run():
    p = json.loads((ROOT / "params" / "f2_params.json").read_text(encoding="utf-8"))
    dt = p["time_grid_dt_s"]
    z, wn = p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"]
    sig = z * wn
    wd = wn * np.sqrt(1 - z ** 2)
    out = {"note": "가상 파라미터 시뮬레이션 결과", "plant": {"zeta": z, "wn_rad_s": wn}}

    # A. 극점과 식
    poles = np.roots([1.0, 2 * sig, wn ** 2])
    t_end = 10.0
    m = extra_metrics(std_tf(z, wn), t_end, dt)
    os_an = 100 * np.exp(-np.pi * z / np.sqrt(1 - z ** 2))
    tp_an = np.pi / wd
    phi = np.arctan2(z, np.sqrt(1 - z ** 2))  # tanφ = ζ/√(1-ζ²)
    out["poles"] = {
        "real": float(poles[0].real), "imag_abs": float(abs(poles[0].imag)),
        "sigma": sig, "wd": float(wd), "theta_deg": float(np.degrees(np.arctan2(wd, sig))),
        "phi_deg": float(np.degrees(phi)),
        "root_check_modulus": float(abs(poles[0])),
    }
    out["formulas"] = {
        "overshoot_analytic_pct": float(os_an), "overshoot_numeric_pct": m["overshoot_pct"],
        "peak_time_analytic_s": float(tp_an), "peak_time_numeric_s": m["peak_time_s"],
        "rise_numeric_s": m["rise_s"], "settle_numeric_s": m["settle_s"],
        "rel_err_overshoot": abs(m["overshoot_pct"] / os_an - 1),
        "rel_err_peak_time": abs(m["peak_time_s"] / tp_an - 1),
    }

    # B. 정정시간 근사 오차 (ωn=11 고정, ζ 스윕)
    rows = []
    for zz in p["zeta_sweep_underdamped"]:
        s_ = zz * wn
        env = -np.log(0.02 * np.sqrt(1 - zz ** 2)) / s_      # 포락선이 2%가 되는 시각
        approx4 = 4.0 / s_
        tend = max(10.0, 8 * approx4)
        mm = extra_metrics(std_tf(zz, wn), tend, dt)
        rows.append({"zeta": zz, "settle_numeric_s": mm["settle_s"], "settle_envelope_s": float(env),
                     "settle_approx4_s": float(approx4),
                     "err_approx4_pct": float(100 * (approx4 / mm["settle_s"] - 1)),
                     "err_envelope_pct": float(100 * (env / mm["settle_s"] - 1)),
                     "overshoot_pct": mm["overshoot_pct"]})
    out["settle_sweep"] = rows

    # C. 영점 추가
    t = np.arange(0.0, 6.0, dt)
    y0 = step_y(std_tf(z, wn), t)
    zero_rows = []
    for r in p["zero_lhp_ratio_to_sigma"]:
        a = r * sig
        mm = extra_metrics(zero_tf(z, wn, a), 6.0, dt)
        zero_rows.append({"kind": "LHP", "ratio_to_sigma": r, "alpha": a, **mm})
    for r in p["zero_rhp_ratio_to_sigma"]:
        a = -r * sig
        mm = extra_metrics(zero_tf(z, wn, a), 6.0, dt)
        zero_rows.append({"kind": "RHP", "ratio_to_sigma": r, "alpha": a, **mm})
    base = extra_metrics(std_tf(z, wn), 6.0, dt)
    out["zero_base"] = {"rise_s": base["rise_s"], "overshoot_pct": base["overshoot_pct"],
                        "peak_time_s": base["peak_time_s"], "settle_s": base["settle_s"]}
    out["zero_sweep"] = zero_rows

    # 항등식 ỹ/α = y + ẏ/α (정규화 영점 응답 = 원 응답 + 미분/α), α=2σ 사례
    a = 2 * sig
    yz = step_y(zero_tf(z, wn, a), t)
    ydot = np.gradient(y0, t)
    out["zero_identity_check"] = {"alpha": a, "max_abs_err": float(np.max(np.abs(yz - (y0 + ydot / a))[5:-5]))}

    # D. 입력 경로: 단위 피드백 루프 L = ωn²/(s(s+2ζωn))
    Ln, Ld = np.array([wn ** 2]), np.polymul([1.0, 0.0], [1.0, 2 * sig])
    Tn, Td = Ln, np.polyadd(Ld, Ln)           # T = L/(1+L)
    Sn, Sd = Ld, np.polyadd(Ld, Ln)           # S = 1/(1+L) = Ld/(Ld+Ln)
    T = signal.TransferFunction(Tn, Td)
    S = signal.TransferFunction(Sn, Sd)
    yT, yS = step_y(T, t), step_y(S, t)
    mT = extra_metrics(T, 6.0, dt)
    iS = int(np.argmin(yS))
    out["input_path"] = {
        "poles_T": [complex(x).real for x in np.roots(Td)], "poles_S_same": bool(np.allclose(np.sort_complex(np.roots(Td)), np.sort_complex(np.roots(Sd)))),
        "T_matches_standard_max_err": float(np.max(np.abs(yT - y0))),
        "S_initial": float(yS[0]), "S_final": float(yS[-1]),
        "S_min": float(yS.min()), "S_min_time_s": float(t[iS]),
        "S_first_zero_cross_s": float(t[int(np.argmax(yS <= 0))]),
        "S_settle_to_2pct_s": float(t[np.where(np.abs(yS) > 0.02)[0][-1] + 1]),
        "T_peak_time_s": mT["peak_time_s"],
        "S_zeros": [0.0, -2 * sig],
    }

    # E. 초기조건 응답: x(0)=1, x'(0)=0, 입력 0
    A = np.array([[0.0, 1.0], [-wn ** 2, -2 * sig]])
    B = np.array([[0.0], [wn ** 2]])
    Cm = np.array([[1.0, 0.0]])
    ss = signal.StateSpace(A, B, Cm, np.zeros((1, 1)))
    _, yic, _ = signal.lsim(ss, U=np.zeros_like(t), T=t, X0=[1.0, 0.0])
    _, yic_v, _ = signal.lsim(ss, U=np.zeros_like(t), T=t, X0=[0.0, wd])
    closed_ic = np.exp(-sig * t) * (np.cos(wd * t) + (sig / wd) * np.sin(wd * t))
    closed_v = np.exp(-sig * t) * np.sin(wd * t)
    out["initial_condition"] = {
        "free_x0_equals_one_minus_step_max_err": float(np.max(np.abs(yic - (1 - y0)))),
        "free_x0_vs_closed_form_max_err": float(np.max(np.abs(yic - closed_ic))),
        "free_v0_vs_closed_form_max_err": float(np.max(np.abs(yic_v - closed_v))),
        "free_x0_first_zero_cross_s": float(t[int(np.argmax(yic <= 0))]),
        "free_x0_peak_after_cross": float(yic.min()),
    }
    # 같은 극점 표: 모드 계수. 스텝 응답 y = 1 - e^{-σt}(cos ωd t + (σ/ωd) sin ωd t)
    out["mode_coeffs"] = {"cos": 1.0, "sin_over_cos": float(sig / wd)}

    res_dir = ROOT / "results" / "F-2"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))

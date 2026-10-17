"""F-3 시간 간격을 줄이면 오차는 얼마나 줄까. 가상 파라미터 시뮬레이션.

계산 내용
 A. 기준 해(닫힌식)를 행렬 지수 함수(정확한 이산화)와 대조
 B. 전진 Euler와 RK4를 같은 간격 목록으로 돌려 최대 오차와 오차 기울기를 구함
 C. 같은 목표 오차에 필요한 간격·단계 수·우변 계산 횟수를 찾음
 D. 전진 Euler의 발산 한계 간격(참고값)
결과는 results/F-3/results.json에 쓴다.
"""
import json
import math
import pathlib

import numpy as np
from scipy.linalg import expm

ROOT = pathlib.Path(__file__).resolve().parents[2]


def make_system(zeta, wn):
    """x'' + 2ζωn x' + ωn² x = ωn² u 를 (x, v=x') 두 상태로 쓴 우변 f."""
    sigma = zeta * wn
    wd = wn * math.sqrt(1.0 - zeta ** 2)

    def f(x, v, u=1.0):
        return v, wn * wn * (u - x) - 2.0 * sigma * v

    return f, sigma, wd


def exact_step(t, sigma, wd):
    """단위 스텝 입력, 초기 상태 0의 정답 y(t)."""
    return 1.0 - np.exp(-sigma * t) * (np.cos(wd * t) + (sigma / wd) * np.sin(wd * t))


def euler(f, n, h):
    """전진 Euler. 한 단계에 우변을 1번 계산한다."""
    x = v = 0.0
    ys = np.empty(n + 1)
    ys[0] = x
    for k in range(n):
        dx, dv = f(x, v)
        x += h * dx
        v += h * dv
        ys[k + 1] = x
    return ys


def rk4(f, n, h):
    """고전 4차 Runge-Kutta. 한 단계에 우변을 4번 계산한다."""
    x = v = 0.0
    ys = np.empty(n + 1)
    ys[0] = x
    for k in range(n):
        k1x, k1v = f(x, v)
        k2x, k2v = f(x + 0.5 * h * k1x, v + 0.5 * h * k1v)
        k3x, k3v = f(x + 0.5 * h * k2x, v + 0.5 * h * k2v)
        k4x, k4v = f(x + h * k3x, v + h * k3v)
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        ys[k + 1] = x
    return ys


METHODS = {"euler": (euler, 1), "rk4": (rk4, 4)}  # (함수, 한 단계당 우변 계산 횟수)


def max_error(name, f, sigma, wd, t_end, n):
    h = t_end / n
    ys = METHODS[name][0](f, n, h)
    t = np.arange(n + 1) * h
    err = np.abs(ys - exact_step(t, sigma, wd))
    return float(err.max()), h


def final_error(name, f, sigma, wd, t_end, n):
    """마지막 시각 한 점의 절대 오차(최대 오차와 비교하는 참고 지표)."""
    h = t_end / n
    ys = METHODS[name][0](f, n, h)
    return float(abs(ys[-1] - exact_step(n * h, sigma, wd)))


def smallest_n(name, f, sigma, wd, t_end, target, n_start=40):
    """max_error <= target 을 처음 만족하는 단계 수 N(정수)을 사다리 + 이분법으로 찾는다."""
    lo, hi = n_start, n_start
    while max_error(name, f, sigma, wd, t_end, hi)[0] > target:
        lo, hi = hi, hi * 2
    if hi == n_start:
        lo = max(2, n_start // 4)
        while max_error(name, f, sigma, wd, t_end, lo)[0] <= target and lo > 2:
            hi, lo = lo, max(2, lo // 2)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if max_error(name, f, sigma, wd, t_end, mid)[0] > target:
            lo = mid
        else:
            hi = mid
    return hi


def run():
    p = json.loads((ROOT / "params" / "f3_params.json").read_text(encoding="utf-8"))
    zeta, wn = p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"]
    t_end = p["t_end_s"]
    f, sigma, wd = make_system(zeta, wn)
    out = {"note": "가상 파라미터 시뮬레이션 결과",
           "plant": {"zeta": zeta, "wn_rad_s": wn, "sigma": sigma, "wd": wd}, "t_end_s": t_end}

    # A. 기준 해 대조: 정확한 이산화 x[k+1] = Φ x[k] + Γ, Φ = e^{A h}, Γ = A^{-1}(Φ - I) B
    A = np.array([[0.0, 1.0], [-wn ** 2, -2.0 * sigma]])
    B = np.array([0.0, wn ** 2])
    n_ref = 3200
    h_ref = t_end / n_ref
    Phi = expm(A * h_ref)
    Gam = np.linalg.solve(A, (Phi - np.eye(2)) @ B)
    x = np.zeros(2)
    ref = [0.0]
    for _ in range(n_ref):
        x = Phi @ x + Gam
        ref.append(x[0])
    t_ref = np.arange(n_ref + 1) * h_ref
    out["reference_check"] = {
        "n_steps": n_ref,
        "closed_form_vs_matrix_exponential_max_diff": float(np.max(np.abs(exact_step(t_ref, sigma, wd) - np.array(ref)))),
    }

    # B. 간격 스윕
    rows = []
    for h in p["sweep_h_s"]:
        n = int(round(t_end / h))
        assert abs(n * h - t_end) < 1e-12, (h, n)
        row = {"h_s": h, "n_steps": n}
        for name in METHODS:
            row[f"{name}_max_err"] = max_error(name, f, sigma, wd, t_end, n)[0]
            row[f"{name}_final_err"] = final_error(name, f, sigma, wd, t_end, n)
        rows.append(row)
    for i, row in enumerate(rows):
        for name in METHODS:
            row[f"{name}_ratio_from_prev"] = (rows[i - 1][f"{name}_max_err"] / row[f"{name}_max_err"]) if i else None
    out["sweep"] = rows
    fits = {}
    lh = np.log10([r["h_s"] for r in rows])
    for name in METHODS:
        le = np.log10([r[f"{name}_max_err"] for r in rows])
        slope, intercept = np.polyfit(lh, le, 1)
        fits[name] = {"slope": float(slope), "log10_C": float(intercept), "C": float(10 ** intercept),
                      "ratios_per_halving": [r[f"{name}_ratio_from_prev"] for r in rows[1:]]}
    out["fits"] = fits
    fits_final = {}
    for name in METHODS:
        le = np.log10([r[f"{name}_final_err"] for r in rows])
        fits_final[name] = {"slope": float(np.polyfit(lh, le, 1)[0])}
    out["fits_final_time_error"] = fits_final

    # C. 같은 목표 오차에 필요한 간격
    trows = []
    for tgt in p["target_errors"]:
        row = {"target_err": tgt}
        for name, (_, evals) in METHODS.items():
            n = smallest_n(name, f, sigma, wd, t_end, tgt)
            e, h = max_error(name, f, sigma, wd, t_end, n)
            row[name] = {"n_steps": n, "h_s": h, "achieved_err": e, "rhs_evals": n * evals}
        row["h_ratio_rk4_over_euler"] = row["rk4"]["h_s"] / row["euler"]["h_s"]
        row["evals_ratio_euler_over_rk4"] = row["euler"]["rhs_evals"] / row["rk4"]["rhs_evals"]
        trows.append(row)
    out["targets"] = trows

    # D. 전진 Euler가 발산하지 않는 간격 한계 |1 + hλ| < 1 → h < 2σ/ωn² (참고값)
    out["euler_stability"] = {"h_limit_s": 2.0 * sigma / wn ** 2,
                              "max_sweep_h_s": max(p["sweep_h_s"])}

    res_dir = ROOT / "results" / "F-3"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))

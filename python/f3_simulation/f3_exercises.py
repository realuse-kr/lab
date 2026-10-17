"""F-3 연습문제 해답 계산. 가상 파라미터 시뮬레이션. 결과는 results/F-3/exercises_results.json에 쓴다."""
import json
import math

import numpy as np

from f3_step_error import ROOT, make_system, exact_step, euler, rk4, max_error


def heun(f, n, h):
    """개선 Euler(Heun). 한 단계에 우변을 2번 계산한다."""
    x = v = 0.0
    ys = np.empty(n + 1)
    ys[0] = x
    for k in range(n):
        k1x, k1v = f(x, v)
        k2x, k2v = f(x + h * k1x, v + h * k1v)
        x += h / 2 * (k1x + k2x)
        v += h / 2 * (k1v + k2v)
        ys[k + 1] = x
    return ys


def main():
    p = json.loads((ROOT / "params" / "f3_params.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "results" / "F-3" / "results.json").read_text(encoding="utf-8"))
    z, wn = p["plant_second_order"]["zeta"], p["plant_second_order"]["wn_rad_s"]
    f, sigma, wd = make_system(z, wn)
    out = {"note": "가상 파라미터 시뮬레이션 결과"}

    # 문제 1: 전진 Euler 발산 한계. 구간을 20 s로 늘려 한계 아래·위 간격을 비교
    h_lim = 2 * sigma / wn ** 2
    rows = []
    for n in (400, 250):  # h = 0.05, 0.08 (20 s)
        h = 20.0 / n
        ys = euler(f, n, h)
        rows.append({"h_s": h, "n_steps": n, "max_abs_x": float(np.abs(ys).max()),
                     "final_abs_err": float(abs(ys[-1] - 1.0))})
    out["ex1_euler_limit"] = {"h_limit_s": h_lim, "t_end_s": 20.0, "rows": rows}

    # 문제 2: 목표 오차 1e-6에 필요한 RK4 단계 수를 마지막 구간의 상수 C로 예측하고 확인
    sw = res["sweep"][-1]
    c = sw["rk4_max_err"] / sw["h_s"] ** 4
    tgt = 1e-6
    h_pred = (tgt / c) ** 0.25
    n_pred = math.ceil(p["t_end_s"] / h_pred)
    e_pred, _ = max_error("rk4", f, sigma, wd, p["t_end_s"], n_pred)
    out["ex2_rk4_predict"] = {"target_err": tgt, "C_from_last_row": c, "h_pred_s": h_pred,
                              "n_pred": n_pred, "achieved_err_at_n_pred": e_pred}

    # 문제 3: 개선 Euler(Heun)의 오차 차수
    errs = []
    for h in p["sweep_h_s"]:
        n = int(round(p["t_end_s"] / h))
        ys = heun(f, n, p["t_end_s"] / n)
        t = np.arange(n + 1) * (p["t_end_s"] / n)
        errs.append(float(np.abs(ys - exact_step(t, sigma, wd)).max()))
    ratios = [errs[i - 1] / errs[i] for i in range(1, len(errs))]
    slope = float(np.polyfit(np.log10(p["sweep_h_s"]), np.log10(errs), 1)[0])
    out["ex3_heun"] = {"max_err": errs, "ratios_per_halving": ratios, "slope": slope}

    (ROOT / "results" / "F-3" / "exercises_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()

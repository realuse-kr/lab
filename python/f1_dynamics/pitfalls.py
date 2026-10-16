"""F-1 함정 박스 수치와 닫힌식 대조. results.json의 pitfalls 항목에 기록한다. 가상 파라미터 시뮬레이션."""
import json
import numpy as np
from scipy import signal
from bandwidth import ROOT, second_order, first_order, bandwidth_rad_s, step_metrics, HALF_POWER_DB

K3 = 10 ** 0.3  # -3 dB 글자 그대로의 이득 제곱비


def closed_form_u(zeta, K):
    """|G|^2 = 1/K 인 (w/wn)^2. K=2는 반전력점, K=10**0.3은 -3 dB."""
    a = 1 - 2 * zeta ** 2
    return a + np.sqrt(a * a + K - 1)


def main():
    res = json.loads((ROOT / "results" / "F-1" / "results.json").read_text(encoding="utf-8"))
    z, wn = 0.32, 11.0
    s2 = second_order(z, wn)
    out = {}

    # 1) -3 dB 닫힌식 대조(일반 K)
    cf3 = wn * np.sqrt(closed_form_u(z, K3))
    cfh = wn * np.sqrt(closed_form_u(z, 2.0))
    out["closed_form"] = {"minus3dB_rad_s": cf3, "half_power_rad_s": cfh,
                          "numeric_minus3dB_rad_s": bandwidth_rad_s(s2, 10 * wn),
                          "numeric_half_power_rad_s": bandwidth_rad_s(s2, 10 * wn, HALF_POWER_DB)}
    sw = []
    for zz in res["sweep"]:
        z0 = zz["zeta"]
        sw.append({"zeta": z0, "cf_minus3dB": wn * np.sqrt(closed_form_u(z0, K3)), "numeric_minus3dB": zz["bw_rad_s"]})
    out["closed_form_sweep_max_rel_err"] = max(abs(r["cf_minus3dB"] / r["numeric_minus3dB"] - 1) for r in sw)

    # 2) 정상값을 마지막 샘플로 잡을 때(1차계, 6 s까지만 시뮬레이션)
    s1 = first_order(1.0)
    t = np.linspace(0, 6, 60001)
    _, y = signal.step(s1, T=t)
    def metrics(yss):
        yn = y / yss
        def cross(level):
            i = int(np.argmax(yn >= level))
            return t[i - 1] + (level - yn[i - 1]) * (t[i] - t[i - 1]) / (yn[i] - yn[i - 1])
        rise = cross(0.9) - cross(0.1)
        outside = np.where(np.abs(yn - 1) > 0.02)[0]
        return rise, float(t[outside[-1] + 1])
    r_dc, s_dc = metrics(1.0)
    r_ls, s_ls = metrics(float(y[-1]))
    out["last_sample_pitfall_first_order_T6"] = {
        "y_last": float(y[-1]), "rise_dc": r_dc, "settle_dc": s_dc, "rise_last": r_ls, "settle_last": s_ls,
        "rise_rel_err": r_ls / r_dc - 1, "settle_rel_err": s_ls / s_dc - 1}

    # 3) 정정 기준 2% 대 5% (ζ=0.32)
    tt = np.arange(0, 10, 1e-5)
    _, yy = signal.step(s2, T=tt)
    def settle(th):
        o = np.where(np.abs(yy - 1) > th)[0]
        return float(tt[o[-1] + 1])
    out["settling_threshold_second_order"] = {"settle_2pct": settle(0.02), "settle_5pct": settle(0.05)}

    res["pitfalls"] = out
    (ROOT / "results" / "F-1" / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
